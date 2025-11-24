import streamlit as st
import numpy as np
from util import developer_info, developer_info_static
from src.plot import confusion_metrix, roc, correlation_matrix_plotly
from src.handle_null_value import contains_missing_value, remove_high_null, fit_imputers, apply_imputers, replace_placeholders_with_nan
from src.preprocess import remove_rows_with_empty_target, remove_duplicates, fit_encoders, apply_encoders
from src.llm_service import decide_fill_null, decide_encode_type, decide_model, decide_target_attribute, decide_test_ratio, decide_balance
from src.pca import decide_pca, fit_pca_pipeline, apply_pca_pipeline
from src.model_service import split_data_early, check_and_balance, fpr_and_tpr, auc, save_model, calculate_f1_score
from src.predictive_model import train_selected_model
from src.util import select_Y, contain_null_attributes_info, separate_fill_null_list, check_all_columns_numeric, non_numeric_columns_and_head, separate_decode_list, get_data_overview, get_selected_models, get_model_name, count_unique, attribute_info, get_balance_info, get_balance_method_name
from sklearn.preprocessing import StandardScaler
from src.reasoning_display import show_reasoning, show_all_reasoning_summary

def update_balance_data():
    st.session_state.balance_data = st.session_state.to_perform_balance

def start_training_model():
    st.session_state["start_training"] = True

def prediction_model_pipeline(DF, API_KEY, GPT_MODEL):
    """
    FIXED VERSION: Classification pipeline with early train/test split to prevent data leakage.
    
    Key changes:
    1. Train/test split happens EARLY (right after target identification and basic cleaning)
    2. All preprocessing (imputation, encoding, PCA) is fitted on TRAINING data only
    3. LLM decisions are based on TRAINING data statistics only
    4. Same transformations are applied to test data
    """
    st.divider()
    st.subheader('Data Overview')
    
    # Store original data
    if 'data_origin' not in st.session_state:
        st.session_state.data_origin = DF
    st.dataframe(st.session_state.data_origin.describe(), width=1200)
    attributes = st.session_state.data_origin.columns.tolist()
    
    # ========================================================================
    # STEP 1: Select Target Variable (Can use full dataset - just identifying name)
    # ========================================================================
    if 'target_selected' not in st.session_state:
        st.session_state.target_selected = False
    
    st.subheader('Target Variable')
    if not st.session_state.target_selected:
        with st.spinner("AI is analyzing the data..."):
            attributes_for_target, types_info_for_target, head_info_for_target = attribute_info(st.session_state.data_origin)
            st.session_state.target_Y = decide_target_attribute(attributes_for_target, types_info_for_target, head_info_for_target, GPT_MODEL, API_KEY)

        if st.session_state.target_Y != -1:
            selected_Y = st.session_state.target_Y
            st.success("Target variable has been selected by the AI!")
            st.write(f'Target attribute selected: :green[**{selected_Y}**]')
            show_reasoning('target_selection', 'Target Variable Selection')
            st.session_state.target_selected = True
        else:
            st.info("AI cannot determine the target variable from the data. Please select the target variable")
            target_col1, target_col2 = st.columns([9, 1])
            with target_col1:
                selected_Y = st.selectbox(
                    label = 'Select the target variable to predict:',
                    options = attributes,
                    index = len(attributes)-1,
                    label_visibility='collapsed'
                )
            with target_col2:
                if st.button("Confirm", type="primary"):
                    st.session_state.target_selected = True
        st.session_state.selected_Y = selected_Y
    else:
        if st.session_state.target_Y != -1:
            st.success("Target variable has been selected by the AI!")
            show_reasoning('target_selection', 'Target Variable Selection')
        st.write(f"Target variable selected: :green[**{st.session_state.selected_Y}**]")

    if st.session_state.target_selected:
        
        # ====================================================================
        # STEP 2: Basic Cleaning (No leakage - just removing obvious bad data)
        # ====================================================================
        if 'df_basic_cleaned' not in st.session_state:
            df_cleaned = st.session_state.data_origin.copy()
            # Replace placeholders with NaN
            df_cleaned = replace_placeholders_with_nan(df_cleaned)
            # Remove rows/columns with excessive nulls
            df_cleaned = remove_high_null(df_cleaned)
            # Remove rows where target is null
            df_cleaned = remove_rows_with_empty_target(df_cleaned, st.session_state.selected_Y)
            st.session_state.df_basic_cleaned = df_cleaned
        
        # ====================================================================
        # STEP 3: EARLY TRAIN/TEST SPLIT - Prevent Data Leakage
        # ====================================================================
        st.subheader('⚠️ Early Data Splitting (Data Leakage Prevention)')
        
        if 'early_split_done' not in st.session_state:
            # Separate features and target
            X_full, Y_full = select_Y(st.session_state.df_basic_cleaned, st.session_state.selected_Y)
            
            # Early split with stratification
            X_train_raw, X_test_raw, Y_train, Y_test = split_data_early(
                X_full, Y_full, 
                test_size=0.2,  # Default 20%, will be adjusted later
                random_state=42,
                stratify=True
            )
            
            # Encode target variable if it's categorical (for model training)
            # FIT encoding on training labels, APPLY to both train and test
            if Y_train.dtype == 'object':
                # Create mapping from training data
                unique_train_labels = Y_train.unique()
                label_to_int_map = {label: idx for idx, label in enumerate(unique_train_labels)}
                st.session_state.target_label_map = label_to_int_map
                st.session_state.int_to_label_map = {idx: label for label, idx in label_to_int_map.items()}
                
                # Apply encoding to both train and test
                Y_train = Y_train.map(label_to_int_map)
                Y_test = Y_test.map(label_to_int_map)
                
                st.info(f"✅ Target variable encoded: {label_to_int_map}")
            
            # Store raw train/test data (BEFORE preprocessing)
            st.session_state.X_train_raw = X_train_raw
            st.session_state.X_test_raw = X_test_raw
            st.session_state.Y_train = Y_train
            st.session_state.Y_test = Y_test
            st.session_state.early_split_done = True
            
            st.success(f"✅ Data split early: {len(X_train_raw)} train, {len(X_test_raw)} test samples")
            st.info("All preprocessing will now be fitted on training data only!")
        else:
            st.success(f"✅ Early split completed: {len(st.session_state.X_train_raw)} train, {len(st.session_state.X_test_raw)} test")
        
        # ====================================================================
        # STEP 4: Handle Missing Values (FIT on train, APPLY to both)
        # ====================================================================
        st.subheader('Handle and Impute Missing Values')
        
        if "contain_null" not in st.session_state:
            st.session_state.contain_null = contains_missing_value(st.session_state.X_train_raw)

        if 'imputers_fitted' not in st.session_state:
            if st.session_state.contain_null:
                with st.status("Processing **missing values** in the data...", expanded=True) as status:
                    st.write("⚠️ Analyzing TRAINING data only...")
                    
                    # Get null info from TRAINING data only
                    train_df_for_null = st.session_state.X_train_raw.copy()
                    attributes, types_info, description_info = contain_null_attributes_info(train_df_for_null)
                    
                    st.write("Large language model analysis (on training data)...")
                    fill_result_dict = decide_fill_null(attributes, types_info, description_info, GPT_MODEL, API_KEY)
                    
                    st.write("Fitting imputers on training data...")
                    mean_list, median_list, mode_list, new_category_list, interpolation_list = separate_fill_null_list(fill_result_dict)
                    
                    # FIT imputers on training data
                    imputers = fit_imputers(
                        st.session_state.X_train_raw,
                        mean_list, median_list, mode_list, 
                        new_category_list, interpolation_list
                    )
                    st.session_state.imputers = imputers
                    
                    # APPLY to both train and test
                    st.write("Applying imputers to training and test data...")
                    st.session_state.X_train_imputed = apply_imputers(st.session_state.X_train_raw, imputers)
                    st.session_state.X_test_imputed = apply_imputers(st.session_state.X_test_raw, imputers)
                    
                    st.session_state.imputers_fitted = True
                    status.update(label='✅ Missing value processing completed (no leakage)!', state="complete", expanded=False)
                
                show_reasoning('null_filling', 'Missing Value Strategy')
                
                # Offer download of imputed training data
                st.download_button(
                    label="Download Imputed Training Data",
                    data=st.session_state.X_train_imputed.to_csv(index=False).encode('utf-8'),
                    file_name="train_imputed.csv",
                    mime='text/csv')
            else:
                st.session_state.X_train_imputed = st.session_state.X_train_raw
                st.session_state.X_test_imputed = st.session_state.X_test_raw
                st.session_state.imputers_fitted = True
                st.success("No missing values detected in training data. Processing skipped.")
        else:
            st.success("✅ Missing value processing completed (fitted on training data only)!")
            if st.session_state.contain_null:
                show_reasoning('null_filling', 'Missing Value Strategy')
                st.download_button(
                    label="Download Imputed Training Data",
                    data=st.session_state.X_train_imputed.to_csv(index=False).encode('utf-8'),
                    file_name="train_imputed.csv",
                    mime='text/csv')
        
        # ====================================================================
        # STEP 5: Data Encoding (FIT on train, APPLY to both)
        # ====================================================================
        st.subheader("Process Data Encoding")
        st.caption("*For considerations of processing time, **NLP features** like **TF-IDF** have not been included in the current pipeline, long text attributes may be dropped.")
        
        if 'all_numeric' not in st.session_state:
            st.session_state.all_numeric = check_all_columns_numeric(st.session_state.X_train_imputed)
        
        if 'encoders_fitted' not in st.session_state:
            if not st.session_state.all_numeric:
                with st.status("Encoding non-numeric data using **numeric mapping** and **one-hot**...", expanded=True) as status:
                    st.write("⚠️ Analyzing TRAINING data only...")
                    
                    # Get non-numeric info from TRAINING data only
                    non_numeric_attributes, non_numeric_head = non_numeric_columns_and_head(st.session_state.X_train_imputed)
                    
                    st.write("Large language model analysis (on training data)...")
                    encode_result_dict = decide_encode_type(non_numeric_attributes, non_numeric_head, GPT_MODEL, API_KEY)
                    
                    st.write("Fitting encoders on training data...")
                    convert_int_cols, one_hot_cols, drop_cols = separate_decode_list(encode_result_dict, st.session_state.selected_Y)
                    
                    # FIT encoders on training data
                    encoders = fit_encoders(st.session_state.X_train_imputed, convert_int_cols, one_hot_cols)
                    st.session_state.encoders = encoders
                    st.session_state.drop_cols = drop_cols
                    
                    # APPLY to both train and test
                    st.write("Applying encoders to training and test data...")
                    st.session_state.X_train_encoded, _ = apply_encoders(st.session_state.X_train_imputed, encoders, drop_cols)
                    st.session_state.X_test_encoded, _ = apply_encoders(st.session_state.X_test_imputed, encoders, drop_cols)
                    
                    st.session_state.encoders_fitted = True
                    status.update(label='✅ Data encoding completed (no leakage)!', state="complete", expanded=False)
                
                show_reasoning('encoding', 'Data Encoding Strategy')
                
                st.download_button(
                    label="Download Encoded Training Data",
                    data=st.session_state.X_train_encoded.to_csv(index=False).encode('utf-8'),
                    file_name="train_encoded.csv",
                    mime='text/csv')
            else:
                st.session_state.X_train_encoded = st.session_state.X_train_imputed
                st.session_state.X_test_encoded = st.session_state.X_test_imputed
                st.session_state.encoders_fitted = True
                st.success("All columns are numeric. Processing skipped.")
        else:
            st.success("✅ Data encoded (fitted on training data only)!")
            if not st.session_state.all_numeric:
                show_reasoning('encoding', 'Data Encoding Strategy')
                st.download_button(
                    label="Download Encoded Training Data",
                    data=st.session_state.X_train_encoded.to_csv(index=False).encode('utf-8'),
                    file_name="train_encoded.csv",
                    mime='text/csv')
        
        # ====================================================================
        # STEP 6: Remove Duplicates (from both train and test independently)
        # ====================================================================
        st.subheader('Remove Duplicate Entities')
        if 'duplicates_removed' not in st.session_state:
            # Remove duplicates and keep track of indices
            X_train_dedup = remove_duplicates(st.session_state.X_train_encoded)
            X_test_dedup = remove_duplicates(st.session_state.X_test_encoded)
            
            # Filter Y_train to match the deduplicated X_train indices
            Y_train_dedup = st.session_state.Y_train.loc[X_train_dedup.index]
            Y_test_dedup = st.session_state.Y_test.loc[X_test_dedup.index]
            
            st.session_state.X_train_dedup = X_train_dedup
            st.session_state.X_test_dedup = X_test_dedup
            st.session_state.Y_train_dedup = Y_train_dedup
            st.session_state.Y_test_dedup = Y_test_dedup
            st.session_state.duplicates_removed = True
        st.info("Duplicate rows removed from train and test independently.")
        
        # Reattach target for correlation matrix visualization
        if 'train_with_target' not in st.session_state:
            train_viz = st.session_state.X_train_dedup.copy()
            
            # Y is already encoded at this point (done during early split)
            train_viz[st.session_state.selected_Y] = st.session_state.Y_train_dedup.values
            
            # Ensure all columns are numeric for correlation matrix
            train_viz = train_viz.select_dtypes(include=[np.number])
            
            st.session_state.train_with_target = train_viz
        
        st.subheader('Correlation Between Attributes (Training Data)')
        st.plotly_chart(correlation_matrix_plotly(st.session_state.train_with_target))
        
        # ====================================================================
        # STEP 7: PCA (FIT on train, APPLY to both)
        # ====================================================================
        st.subheader('Principal Component Analysis')
        st.write("Deciding whether to perform PCA (based on training data variance)...")
        
        if 'pca_decided' not in st.session_state:
            # Decide PCA based on TRAINING data only
            to_perform_pca, n_components = decide_pca(st.session_state.X_train_dedup)
            st.session_state.to_perform_pca = to_perform_pca
            st.session_state.n_components = n_components
            st.session_state.pca_decided = True
        
        if 'pca_applied' not in st.session_state:
            if st.session_state.to_perform_pca:
                # FIT PCA on training data
                scaler, pca, feature_cols = fit_pca_pipeline(
                    st.session_state.X_train_dedup, 
                    st.session_state.n_components
                )
                st.session_state.pca_transformer = pca
                st.session_state.pca_scaler = scaler
                st.session_state.pca_features = feature_cols
                
                # APPLY to both train and test
                st.session_state.X_train_final = apply_pca_pipeline(
                    st.session_state.X_train_dedup, 
                    scaler, pca, feature_cols
                )
                st.session_state.X_test_final = apply_pca_pipeline(
                    st.session_state.X_test_dedup, 
                    scaler, pca, feature_cols
                )
                st.session_state.pca_performed = True
            else:
                # No PCA - just copy the data
                st.session_state.X_train_final = st.session_state.X_train_dedup
                st.session_state.X_test_final = st.session_state.X_test_dedup
                st.session_state.pca_performed = False
            
            st.session_state.pca_applied = True
        
        st.success(f"✅ PCA {'performed' if st.session_state.pca_performed else 'skipped'} (fitted on training data only)!")
        
        # ====================================================================
        # STEP 8: Determine Test Ratio and Class Balancing Strategy
        # ====================================================================
        if 'balance_data' not in st.session_state:
            st.session_state.balance_data = True
        if "start_training" not in st.session_state:
            st.session_state["start_training"] = False
        if 'model_trained' not in st.session_state:
            st.session_state['model_trained'] = False
        if 'is_binary' not in st.session_state:
            st.session_state['is_binary'] = st.session_state.Y_train_dedup.nunique() == 2

        # AI decide the testing set percentage (based on training data shape)
        if 'test_percentage' not in st.session_state:
            with st.spinner("Deciding testing set percentage based on training data..."):
                # Use training data shape for decision
                train_shape = st.session_state.X_train_final.shape
                st.session_state.test_percentage = int(decide_test_ratio(train_shape, GPT_MODEL, API_KEY) * 100)
            show_reasoning('test_ratio', 'Train-Test Split Strategy')

        splitting_column, balance_column = st.columns(2)
        with splitting_column:
            st.subheader('Data Splitting')
            st.caption('AI recommended test percentage (already split early)')
            st.slider('Percentage of test set', 1, 25, st.session_state.test_percentage, 
                     key='test_percentage_display', disabled=True)
            st.info("Note: Data was split early to prevent leakage. This is for display only.")
        
        with balance_column:
            st.metric(label="Test Data", value=f"{st.session_state.test_percentage}%", delta=None)
            st.toggle('Class Balancing', value=st.session_state.balance_data, key='to_perform_balance', 
                     on_change=update_balance_data, disabled=st.session_state['start_training'])
            st.caption('Strategies for handling imbalanced data sets.')
            st.caption('AI will select the most appropriate method to balance the data.')
        
        st.button("Start Training Model", on_click=start_training_model, type="primary", disabled=st.session_state['start_training'])

        # ====================================================================
        # STEP 9: Model Training
        # ====================================================================
        if st.session_state['start_training']:
            with st.container():
                st.header("Modeling")
                
                # Class Balancing (only on TRAINING data)
                if 'balanced_data' not in st.session_state:
                    if st.session_state.balance_data:
                        with st.spinner("AI is deciding the balance strategy (analyzing training data)..."):
                            # Reattach target temporarily for balance analysis
                            train_for_balance = st.session_state.X_train_final.copy()
                            train_for_balance[st.session_state.selected_Y] = st.session_state.Y_train_dedup.values
                            
                            shape_info_balance, description_info_balance, balance_info_balance = get_balance_info(
                                train_for_balance, st.session_state.selected_Y
                            )
                            st.session_state.balance_method = int(decide_balance(
                                shape_info_balance, description_info_balance, balance_info_balance, GPT_MODEL, API_KEY
                            ))
                            
                            # Balance TRAINING data only (using deduplicated Y_train)
                            X_train_balanced, Y_train_balanced = check_and_balance(
                                st.session_state.X_train_final, 
                                st.session_state.Y_train_dedup, 
                                method=st.session_state.balance_method
                            )
                            st.session_state.X_train_balanced = X_train_balanced
                            st.session_state.Y_train_balanced = Y_train_balanced
                        show_reasoning('balance_strategy', 'Class Balancing Strategy')
                    else:
                        st.session_state.balance_method = 4  # No balancing
                        st.session_state.X_train_balanced = st.session_state.X_train_final
                        st.session_state.Y_train_balanced = st.session_state.Y_train_dedup
                    
                    st.session_state.balanced_data = True
                
                # Standardize if PCA wasn't performed (FIT on train, APPLY to both)
                if 'final_scaling_done' not in st.session_state:
                    if not st.session_state.pca_performed:
                        # Fit scaler on TRAINING data
                        scaler = StandardScaler()
                        st.session_state.X_train_scaled = scaler.fit_transform(st.session_state.X_train_balanced)
                        # Apply to TEST data
                        st.session_state.X_test_scaled = scaler.transform(st.session_state.X_test_final)
                    else:
                        # PCA already standardized
                        st.session_state.X_train_scaled = st.session_state.X_train_balanced.values
                        st.session_state.X_test_scaled = st.session_state.X_test_final.values
                    
                    st.session_state.final_scaling_done = True
                
                # Decide model types (based on training data)
                if "decided_model" not in st.session_state:
                    with st.spinner("Deciding models based on training data..."):
                        # Reattach target for model decision (use deduplicated Y_train)
                        train_for_model = st.session_state.X_train_final.copy()
                        train_for_model[st.session_state.selected_Y] = st.session_state.Y_train_dedup.values
                        
                        shape_info, head_info, nunique_info, description_info = get_data_overview(train_for_model)
                        model_dict = decide_model(shape_info, head_info, nunique_info, description_info, GPT_MODEL, API_KEY)
                        model_list = get_selected_models(model_dict)
                        st.session_state.model_list = model_list
                        st.session_state["decided_model"] = True
                    show_reasoning('model_selection', 'Model Selection Strategy')

                # Display results
                if st.session_state["decided_model"]:
                    display_results(
                        st.session_state.X_train_scaled, 
                        st.session_state.X_test_scaled, 
                        st.session_state.Y_train_balanced, 
                        st.session_state.Y_test_dedup
                    )
                    st.session_state["all_set"] = True
                
                # Download models
                if st.session_state.get("all_set", False):
                    download_col1, download_col2, download_col3 = st.columns(3)
                    with download_col1:
                        st.download_button(label="Download Model", data=st.session_state.downloadable_model1, file_name=f"{st.session_state.model1_name}.joblib", mime="application/octet-stream")
                    with download_col2:
                        st.download_button(label="Download Model", data=st.session_state.downloadable_model2, file_name=f"{st.session_state.model2_name}.joblib", mime="application/octet-stream")
                    with download_col3:
                        st.download_button(label="Download Model", data=st.session_state.downloadable_model3, file_name=f"{st.session_state.model3_name}.joblib", mime="application/octet-stream")

        # Footer
        st.divider()
        if "all_set" in st.session_state and st.session_state["all_set"]:
            show_all_reasoning_summary()
            if "has_been_set" not in st.session_state:
                st.session_state["has_been_set"] = True
                developer_info()
            else:
                developer_info_static()

def display_results(X_train, X_test, Y_train, Y_test):
    st.success("Models selected based on your data!")
    
    # Data set metrics
    data_col1, data_col2, data_col3, balance_col4 = st.columns(4)
    with data_col1:
        st.metric(label="Total Data", value=len(X_train)+len(X_test), delta=None)
    with data_col2:
        st.metric(label="Training Data", value=len(X_train), delta=None)
    with data_col3:
        st.metric(label="Testing Data", value=len(X_test), delta=None)
    with balance_col4:
        st.metric(label="Balance Strategy", value=get_balance_method_name(st.session_state.balance_method), delta=None)
    
    # Model training
    model_col1, model_col2, model_col3 = st.columns(3)
    with model_col1:
        if "model1_name" not in st.session_state:
            st.session_state.model1_name = get_model_name(st.session_state.model_list[0])
        st.subheader(st.session_state.model1_name)
        with st.spinner("Model training in progress..."):
            if 'model1' not in st.session_state:
                st.session_state.model1 = train_selected_model(X_train, Y_train, st.session_state.model_list[0])
                st.session_state.downloadable_model1 = save_model(st.session_state.model1)
        # Model metrics
        st.write(f"The accuracy of the {st.session_state.model1_name}: ", f'\n:green[**{st.session_state.model1.score(X_test, Y_test)}**]')
        st.pyplot(confusion_metrix(st.session_state.model1_name, st.session_state.model1, X_test, Y_test))
        st.write("F1 Score: ", f':green[**{calculate_f1_score(st.session_state.model1, X_test, Y_test, st.session_state.is_binary)}**]')
        if st.session_state.model_list[0] != 2 and st.session_state['is_binary']:
            if 'fpr1' not in st.session_state:
                fpr1, tpr1 = fpr_and_tpr(st.session_state.model1, X_test, Y_test)
                st.session_state.fpr1 = fpr1
                st.session_state.tpr1 = tpr1
            st.pyplot(roc(st.session_state.model1_name, st.session_state.fpr1, st.session_state.tpr1))
            st.write(f"The AUC of the {st.session_state.model1_name}: ", f'\n:green[**{auc(st.session_state.fpr1, st.session_state.tpr1)}**]')

    with model_col2:
        if "model2_name" not in st.session_state:
            st.session_state.model2_name = get_model_name(st.session_state.model_list[1])
        st.subheader(st.session_state.model2_name)
        with st.spinner("Model training in progress..."):
            if 'model2' not in st.session_state:
                st.session_state.model2 = train_selected_model(X_train, Y_train, st.session_state.model_list[1])
                st.session_state.downloadable_model2 = save_model(st.session_state.model2)
        # Model metrics
        st.write(f"The accuracy of the {st.session_state.model2_name}: ", f'\n:green[**{st.session_state.model2.score(X_test, Y_test)}**]')
        st.pyplot(confusion_metrix(st.session_state.model2_name, st.session_state.model2, X_test, Y_test))
        st.write("F1 Score: ", f':green[**{calculate_f1_score(st.session_state.model2, X_test, Y_test, st.session_state.is_binary)}**]')
        if st.session_state.model_list[1] != 2 and st.session_state['is_binary']:
            if 'fpr2' not in st.session_state:
                fpr2, tpr2 = fpr_and_tpr(st.session_state.model2, X_test, Y_test)
                st.session_state.fpr2 = fpr2
                st.session_state.tpr2 = tpr2
            st.pyplot(roc(st.session_state.model2_name, st.session_state.fpr2, st.session_state.tpr2))
            st.write(f"The AUC of the {st.session_state.model2_name}: ", f'\n:green[**{auc(st.session_state.fpr2, st.session_state.tpr2)}**]')
        
    with model_col3:
        if "model3_name" not in st.session_state:
            st.session_state.model3_name = get_model_name(st.session_state.model_list[2])
        st.subheader(st.session_state.model3_name)
        with st.spinner("Model training in progress..."):
            if 'model3' not in st.session_state:
                st.session_state.model3 = train_selected_model(X_train, Y_train, st.session_state.model_list[2])
                st.session_state.downloadable_model3 = save_model(st.session_state.model3)
        # Model metrics
        st.write(f"The accuracy of the {st.session_state.model3_name}: ", f'\n:green[**{st.session_state.model3.score(X_test, Y_test)}**]')
        st.pyplot(confusion_metrix(st.session_state.model3_name, st.session_state.model3, X_test, Y_test))
        st.write("F1 Score: ", f':green[**{calculate_f1_score(st.session_state.model3, X_test, Y_test, st.session_state.is_binary)}**]')
        if st.session_state.model_list[2] != 2 and st.session_state['is_binary']:
            if 'fpr3' not in st.session_state:
                fpr3, tpr3 = fpr_and_tpr(st.session_state.model3, X_test, Y_test)
                st.session_state.fpr3 = fpr3
                st.session_state.tpr3 = tpr3
            st.pyplot(roc(st.session_state.model3_name, st.session_state.fpr3, st.session_state.tpr3))
            st.write(f"The AUC of the {st.session_state.model3_name}: ", f'\n:green[**{auc(st.session_state.fpr3, st.session_state.tpr3)}**]')

