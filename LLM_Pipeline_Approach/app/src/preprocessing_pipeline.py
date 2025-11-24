"""
Preprocessing Pipeline to Prevent Data Leakage

This module provides a unified preprocessing pipeline that ensures all transformers
are fitted on training data only and applied consistently to both training and test data.

The pipeline handles:
- Null value imputation
- Categorical encoding (integer mapping and one-hot)
- PCA dimensionality reduction
- Data standardization

All transformers are fitted on training data and stored for later application to test data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class PreprocessingPipeline:
    """
    Manages all preprocessing steps to prevent data leakage.
    All transformers are fitted on training data only.
    """
    
    def __init__(self):
        # Store fitted imputers for each column
        self.null_imputers = {}
        
        # Store fitted encoders for each column
        self.encoders = {}
        
        # Store fitted scaler (if not using PCA)
        self.scaler = None
        
        # Store fitted PCA transformer and related info
        self.pca = None
        self.pca_scaler = None
        self.pca_feature_columns = None
        
        # Store columns to drop
        self.drop_columns = []
        
        # Track if pipeline has been fitted
        self.fitted = False
        
        # Store target column name
        self.target_column = None
        
    def fit_null_imputers(self, df_train, mean_list, median_list, mode_list, 
                         new_category_list, interpolation_list):
        """
        Fit imputers on training data and store them.
        
        Parameters:
        - df_train: Training DataFrame
        - mean_list: Columns to impute with mean
        - median_list: Columns to impute with median
        - mode_list: Columns to impute with mode
        - new_category_list: Columns to impute with new category (will use most frequent)
        - interpolation_list: Columns to impute with interpolation (will use median as fallback)
        
        Returns: None (stores imputers internally)
        """
        # Fit mean imputers
        for col in mean_list:
            if col in df_train.columns:
                self.null_imputers[col] = {
                    'method': 'mean',
                    'value': df_train[col].mean()
                }
        
        # Fit median imputers
        for col in median_list:
            if col in df_train.columns:
                self.null_imputers[col] = {
                    'method': 'median',
                    'value': df_train[col].median()
                }
        
        # Fit mode imputers
        for col in mode_list:
            if col in df_train.columns:
                mode_values = df_train[col].mode()
                if not mode_values.empty:
                    self.null_imputers[col] = {
                        'method': 'mode',
                        'value': mode_values[0]
                    }
        
        # For new_category_list, use mode as well (most frequent value)
        for col in new_category_list:
            if col in df_train.columns:
                mode_values = df_train[col].mode()
                if not mode_values.empty:
                    self.null_imputers[col] = {
                        'method': 'mode',
                        'value': mode_values[0]
                    }
        
        # For interpolation_list, use median as a safe fallback for unseen data
        for col in interpolation_list:
            if col in df_train.columns:
                self.null_imputers[col] = {
                    'method': 'median',
                    'value': df_train[col].median()
                }
    
    def apply_null_imputers(self, df):
        """
        Apply fitted imputers to any dataframe.
        
        Parameters:
        - df: DataFrame to impute
        
        Returns: Imputed DataFrame
        """
        df_copy = df.copy()
        
        for col, imputer in self.null_imputers.items():
            if col in df_copy.columns:
                if imputer['method'] in ['mean', 'median', 'mode']:
                    df_copy[col] = df_copy[col].fillna(imputer['value'])
        
        return df_copy
    
    def fit_encoders(self, df_train, convert_int_cols, one_hot_cols, drop_cols):
        """
        Fit encoders on training data and store them.
        
        Parameters:
        - df_train: Training DataFrame
        - convert_int_cols: Columns to encode with integer mapping
        - one_hot_cols: Columns to encode with one-hot
        - drop_cols: Columns to drop
        
        Returns: None (stores encoders internally)
        """
        self.drop_columns = drop_cols
        
        # Fit integer encoders
        for col in convert_int_cols:
            if col in df_train.columns and df_train[col].dtype == 'object':
                unique_values = df_train[col].unique()
                # Filter out NaN values
                unique_values = [val for val in unique_values if pd.notna(val)]
                self.encoders[col] = {
                    'type': 'integer',
                    'mapping': {val: idx for idx, val in enumerate(unique_values)}
                }
        
        # Fit one-hot encoders
        for col in one_hot_cols:
            if col in df_train.columns:
                if df_train[col].dtype == 'object' or df_train[col].dtype == 'category':
                    unique_values = df_train[col].unique()
                    # Filter out NaN values
                    unique_values = [val for val in unique_values if pd.notna(val)]
                    self.encoders[col] = {
                        'type': 'onehot',
                        'categories': unique_values
                    }
    
    def apply_encoders(self, df):
        """
        Apply fitted encoders to any dataframe.
        Handles unseen categories gracefully.
        
        Parameters:
        - df: DataFrame to encode
        
        Returns: Encoded DataFrame
        """
        df_copy = df.copy()
        
        for col, encoder in self.encoders.items():
            if col not in df_copy.columns:
                continue
                
            if encoder['type'] == 'integer':
                # Map known values, use -1 for unknown
                df_copy[col] = df_copy[col].map(encoder['mapping']).fillna(-1).astype(int)
            
            elif encoder['type'] == 'onehot':
                # Create columns for known categories only
                for category in encoder['categories']:
                    new_col_name = f"{col}_{category}"
                    df_copy[new_col_name] = (df_copy[col] == category).astype(int)
                
                # Drop the original column
                df_copy = df_copy.drop(col, axis=1)
        
        # Drop specified columns
        df_copy = df_copy.drop(columns=self.drop_columns, errors='ignore')
        
        return df_copy
    
    def fit_pca(self, df_train, n_components, target_column=None):
        """
        Fit StandardScaler and PCA on training data.
        
        Parameters:
        - df_train: Training DataFrame
        - n_components: Number of principal components to keep
        - target_column: Name of target column to exclude from PCA
        
        Returns: None (stores PCA and scaler internally)
        """
        self.target_column = target_column
        
        # Separate features from target
        if target_column and target_column in df_train.columns:
            X_train = df_train.drop(target_column, axis=1)
        else:
            X_train = df_train
        
        # Only use numeric columns
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        self.pca_feature_columns = numeric_cols
        
        # Fit scaler
        self.pca_scaler = StandardScaler()
        X_scaled = self.pca_scaler.fit_transform(X_train[numeric_cols])
        
        # Fit PCA
        self.pca = PCA(n_components=n_components)
        self.pca.fit(X_scaled)
    
    def apply_pca(self, df):
        """
        Apply fitted scaler and PCA to any dataframe.
        
        Parameters:
        - df: DataFrame to transform
        
        Returns: DataFrame with principal components
        """
        # Extract target if present
        target_data = None
        if self.target_column and self.target_column in df.columns:
            target_data = df[self.target_column].copy()
            X = df.drop(self.target_column, axis=1)
        else:
            X = df
        
        # Transform using fitted scaler and PCA
        X_scaled = self.pca_scaler.transform(X[self.pca_feature_columns])
        X_pca = self.pca.transform(X_scaled)
        
        # Create result dataframe
        pca_df = pd.DataFrame(
            X_pca,
            columns=[f'PC{i+1}' for i in range(self.pca.n_components_)],
            index=df.index
        )
        
        # Reattach target
        if target_data is not None:
            pca_df[self.target_column] = target_data.values
        
        return pca_df
    
    def fit_scaler(self, df_train):
        """
        Fit StandardScaler on training data (used when PCA is not performed).
        
        Parameters:
        - df_train: Training DataFrame (features only, no target)
        
        Returns: None (stores scaler internally)
        """
        self.scaler = StandardScaler()
        self.scaler.fit(df_train)
    
    def apply_scaler(self, df):
        """
        Apply fitted StandardScaler to any dataframe.
        
        Parameters:
        - df: DataFrame to scale
        
        Returns: Scaled array
        """
        return self.scaler.transform(df)
    
    def fit(self, X_train_raw, preprocessing_decisions):
        """
        Fit all preprocessing steps on training data.
        
        Parameters:
        - X_train_raw: Raw training features (DataFrame)
        - preprocessing_decisions: Dictionary containing:
            - 'null_imputation': dict with mean_list, median_list, mode_list, etc.
            - 'encoding': dict with convert_int_cols, one_hot_cols, drop_cols
            - 'pca': dict with perform_pca (bool), n_components (int), target_column (str)
            - 'scaling': bool indicating if scaling should be done (when PCA is False)
        
        Returns: Transformed training data
        """
        df_train = X_train_raw.copy()
        
        # Step 1: Fit and apply null imputation
        if 'null_imputation' in preprocessing_decisions:
            null_config = preprocessing_decisions['null_imputation']
            self.fit_null_imputers(
                df_train,
                null_config.get('mean_list', []),
                null_config.get('median_list', []),
                null_config.get('mode_list', []),
                null_config.get('new_category_list', []),
                null_config.get('interpolation_list', [])
            )
            df_train = self.apply_null_imputers(df_train)
        
        # Step 2: Fit and apply encoding
        if 'encoding' in preprocessing_decisions:
            enc_config = preprocessing_decisions['encoding']
            self.fit_encoders(
                df_train,
                enc_config.get('convert_int_cols', []),
                enc_config.get('one_hot_cols', []),
                enc_config.get('drop_cols', [])
            )
            df_train = self.apply_encoders(df_train)
        
        self.fitted = True
        return df_train
    
    def transform(self, X_raw):
        """
        Apply all fitted transformers to any dataset.
        
        Parameters:
        - X_raw: Raw features (DataFrame)
        
        Returns: Transformed data
        """
        if not self.fitted:
            raise ValueError("Pipeline must be fitted before transform. Call fit() first.")
        
        df = X_raw.copy()
        
        # Apply null imputation
        if self.null_imputers:
            df = self.apply_null_imputers(df)
        
        # Apply encoding
        if self.encoders or self.drop_columns:
            df = self.apply_encoders(df)
        
        return df
    
    def fit_transform(self, X_train_raw, preprocessing_decisions):
        """
        Fit on training data and transform it.
        
        Parameters:
        - X_train_raw: Raw training features
        - preprocessing_decisions: Dictionary with preprocessing configuration
        
        Returns: Transformed training data
        """
        return self.fit(X_train_raw, preprocessing_decisions)

