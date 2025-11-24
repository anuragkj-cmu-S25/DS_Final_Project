import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, PowerTransformer

def convert_to_numeric(df, convert_int_cols_list, one_hot_cols_list, drop_cols):
    """
    Convert specified columns in the DataFrame to numeric formats and drop specified columns.
    Integer conversion and one-hot encoding are applied based on the provided lists of columns.
    Returns a modified DataFrame and a dictionary of mappings used for conversions.

    :param df: Pandas DataFrame to be processed.
    :param convert_int_cols_list: List of column names to be converted to integer type.
    :param one_hot_cols_list: List of column names to be converted to one-hot encoding.
    :param drop_cols: List of column names to be dropped from the DataFrame.
    :return: A tuple with two elements:
             1. DataFrame with specified columns converted and specified columns dropped.
             2. Dictionary of mappings for each conversion type ('integer_mappings' and 'one_hot_mappings').
    """
    df, int_mapping = convert_to_integer(df, convert_int_cols_list)
    df, one_hot_mapping = convert_to_one_hot(df, one_hot_cols_list)
    df = df.drop(columns=drop_cols, errors='ignore')
    mappings = {'integer_mappings': int_mapping, 'one_hot_mappings': one_hot_mapping}
    return df, mappings

def convert_to_integer(df, columns_to_convert=[]):
    """
    Convert specified non-numeric columns in the DataFrame to integer type,
    and return a dictionary of mappings from original values to integers.

    :param df: Pandas DataFrame to be processed.
    :param columns_to_convert: List of column names to be converted to integer type.
    :return: A tuple with two elements:
             1. DataFrame with specified columns converted to integer type.
             2. Dictionary of mappings for each converted column.
    """
    mappings = {}
    for column in columns_to_convert:

        if df[column].dtype == 'object':
            # Create a mapping from unique values to integers
            unique_values = df[column].unique()
            int_to_value_map = {i: value for i, value in enumerate(unique_values)}
            mappings[column] = int_to_value_map

            # Apply the reversed mapping to the DataFrame
            value_to_int_map = {v: k for k, v in int_to_value_map.items()}
            df[column] = df[column].map(value_to_int_map)

    return df, mappings

def convert_to_one_hot(df, columns_to_convert=[]):
    """
    Convert specified non-numeric columns in the DataFrame to one-hot encoding,
    and return a modified DataFrame and a dictionary of mappings used for one-hot encoding.

    :param df: Pandas DataFrame to be processed.
    :param columns_to_convert: List of column names to be converted to one-hot encoding.
    :return: A tuple with two elements:
             1. DataFrame with specified columns converted to one-hot encoding.
             2. Dictionary of mappings for each converted column.
    """
    mappings = {}
    df_modified = df.copy()

    for column in columns_to_convert:
        # Check if the column is categorical
        if df[column].dtype == 'object' or df[column].dtype == 'category':
            # Perform one-hot encoding
            one_hot = pd.get_dummies(df[column], prefix=column)
            # Add the new columns to the modified DataFrame
            df_modified = pd.concat([df_modified, one_hot], axis=1)
            # Drop the original column
            df_modified = df_modified.drop(column, axis=1)

            # Store the mapping
            mappings[column] = {i: column + '_' + str(i) for i in df[column].unique()}

    return df_modified, mappings

def remove_rows_with_empty_target(df, Y_name):
    """
    Remove rows from the DataFrame where the target column has empty values.

    :param df: Pandas DataFrame to be processed.
    :param Y_name: Name of the target column to check for empty values.
    :return: DataFrame with rows removed where target column value is empty.
    """
    # Remove rows where the target column is empty (NaN)
    cleaned_df = df.dropna(subset=[Y_name])
    return cleaned_df

def remove_duplicates(df):
    """
    Remove duplicate rows from the DataFrame.
    """
    return df.drop_duplicates()

def transform_data_for_clustering(df):
    """
    Transform numeric columns in the DataFrame for clustering.
    Applies a PowerTransformer to columns with skewness over a threshold and standardizes them.
    This can help in making the clustering algorithm more effective by normalizing the scale of numerical features.

    :param df: Pandas DataFrame to be transformed.
    :return: DataFrame with transformed numeric columns suitable for clustering.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    transformed_df = df.copy()
    pt = PowerTransformer(method='box-cox', standardize=False)

    for col in numeric_cols:
        if (transformed_df[col] > 0).all():
            skewness = stats.skew(transformed_df[col])
            if abs(skewness) > 0.5:
                transformed_data = pt.fit_transform(transformed_df[[col]])
                transformed_df[col] = transformed_data

    scaler = StandardScaler()
    transformed_df[numeric_cols] = scaler.fit_transform(transformed_df[numeric_cols])
    
    return transformed_df


# ============================================================================
# NEW: Fit/Apply Pattern for Preventing Data Leakage
# ============================================================================

def fit_encoders(df_train, convert_int_cols, one_hot_cols):
    """
    Fit encoders on training data and return them.
    This ensures encoding schemes are learned from training data only.
    
    Parameters:
    - df_train: Training DataFrame
    - convert_int_cols: List of columns to encode with integer mapping
    - one_hot_cols: List of columns to encode with one-hot
    
    Returns:
    - encoders: Dictionary of encoder configurations
    """
    encoders = {}
    
    # Fit integer encoders
    for col in convert_int_cols:
        if col in df_train.columns and df_train[col].dtype == 'object':
            unique_values = df_train[col].unique()
            # Filter out NaN values
            unique_values = [val for val in unique_values if pd.notna(val)]
            encoders[col] = {
                'type': 'integer',
                'mapping': {val: idx for idx, val in enumerate(unique_values)},
                'int_to_value': {idx: val for idx, val in enumerate(unique_values)}
            }
    
    # Fit one-hot encoders
    for col in one_hot_cols:
        if col in df_train.columns:
            if df_train[col].dtype == 'object' or df_train[col].dtype == 'category':
                unique_values = df_train[col].unique()
                # Filter out NaN values
                unique_values = [val for val in unique_values if pd.notna(val)]
                encoders[col] = {
                    'type': 'onehot',
                    'categories': list(unique_values)
                }
    
    return encoders


def apply_encoders(df, encoders, drop_cols=None):
    """
    Apply fitted encoders to any dataframe.
    Handles unseen categories gracefully (maps to -1 for integer encoding,
    creates all-zero column for one-hot encoding).
    
    Parameters:
    - df: DataFrame to encode
    - encoders: Dictionary of fitted encoders from fit_encoders()
    - drop_cols: List of columns to drop (optional)
    
    Returns:
    - df_encoded: Encoded DataFrame
    - mappings: Dictionary of mappings (for compatibility with existing code)
    """
    df_copy = df.copy()
    mappings = {'integer_mappings': {}, 'one_hot_mappings': {}}
    
    for col, encoder in encoders.items():
        if col not in df_copy.columns:
            continue
        
        if encoder['type'] == 'integer':
            # Map known values, use -1 for unknown
            df_copy[col] = df_copy[col].map(encoder['mapping']).fillna(-1).astype(int)
            # Store mapping for return
            mappings['integer_mappings'][col] = encoder['int_to_value']
        
        elif encoder['type'] == 'onehot':
            # Create columns for known categories only
            one_hot_mapping = {}
            for category in encoder['categories']:
                new_col_name = f"{col}_{category}"
                df_copy[new_col_name] = (df_copy[col] == category).astype(int)
                one_hot_mapping[category] = new_col_name
            
            # Drop the original column
            df_copy = df_copy.drop(col, axis=1)
            mappings['one_hot_mappings'][col] = one_hot_mapping
    
    # Drop specified columns
    if drop_cols:
        df_copy = df_copy.drop(columns=drop_cols, errors='ignore')
    
    return df_copy, mappings
