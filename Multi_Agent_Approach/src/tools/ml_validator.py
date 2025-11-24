import ast
import re
from typing import Annotated
from langchain_core.tools import tool
from ..logger import setup_logger

logger = setup_logger()

@tool
def validate_ml_code(
    code: Annotated[str, "Python code to validate for data leakage"]
) -> Annotated[dict, "Validation results"]:
    """
    Validate ML code for common data leakage patterns.
    Returns warnings and suggestions.
    """
    
    warnings = []
    suggestions = []
    
    # Pattern 1: fit_transform on full dataset before split
    pattern1 = r'\.fit_transform\([^)]+\).*train_test_split'
    if re.search(pattern1, code, re.DOTALL):
        warnings.append("⚠️ LEAKAGE: fit_transform detected before train_test_split")
        suggestions.append("Split data first, then fit_transform on training data only")
    
    # Pattern 2: Feature selection before split
    pattern2 = r'SelectKBest.*\.fit_transform.*train_test_split'
    if re.search(pattern2, code, re.DOTALL):
        warnings.append("⚠️ LEAKAGE: Feature selection before train/test split")
        suggestions.append("Use Pipeline or select features after splitting")
    
    # Pattern 3: Group statistics with target variable
    pattern3 = r'\.groupby\([^)]+\)\[.*["\']target["\']\]\.transform'
    if re.search(pattern3, code):
        warnings.append("⚠️ LEAKAGE: Target encoding without proper isolation")
        suggestions.append("Use TargetEncoder in a Pipeline with cross-validation")
    
    # Pattern 4: Random split for time series
    has_time_features = any(word in code.lower() for word in ['date', 'time', 'timestamp', 'datetime'])
    has_random_split = 'train_test_split' in code and 'shuffle=False' not in code
    if has_time_features and has_random_split:
        warnings.append("⚠️ TEMPORAL LEAKAGE: Random split detected for time series data")
        suggestions.append("Use TimeSeriesSplit or temporal train/test split")
    
    # Pattern 5: No pipeline usage
    has_preprocessing = any(word in code for word in ['StandardScaler', 'MinMaxScaler', 'Normalizer', 'SelectKBest'])
    has_model = any(word in code for word in ['fit(', 'RandomForest', 'LogisticRegression', 'SVC'])
    has_pipeline = 'Pipeline' in code
    
    if has_preprocessing and has_model and not has_pipeline:
        warnings.append("⚠️ RISK: Preprocessing and modeling without Pipeline")
        suggestions.append("Use sklearn Pipeline to ensure correct preprocessing order")
    
    # Pattern 6: fit() called on test data
    pattern6 = r'test.*\.fit\('
    if re.search(pattern6, code):
        warnings.append("🔴 CRITICAL LEAKAGE: fit() called on test data")
        suggestions.append("NEVER fit on test data - use transform() only")
    
    # Check for good practices
    good_practices = []
    if 'Pipeline' in code:
        good_practices.append("✅ Using Pipeline - good practice")
    if 'stratify=' in code:
        good_practices.append("✅ Using stratified split - good practice")
    if 'random_state=' in code:
        good_practices.append("✅ Setting random_state - reproducible")
    if 'cross_val_score' in code:
        good_practices.append("✅ Using cross-validation - rigorous evaluation")
    
    return {
        "is_valid": len(warnings) == 0,
        "warnings": warnings,
        "suggestions": suggestions,
        "good_practices": good_practices,
        "severity": "CRITICAL" if any("CRITICAL" in w for w in warnings) else 
                   "HIGH" if warnings else "OK"
    }