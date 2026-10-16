import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Define feature subsets based on the Credit Risk Dataset
NUMERICAL_FEATURES = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length"
]

CATEGORICAL_FEATURES = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file"
]

TARGET_COLUMN = "loan_status"

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw credit risk data by:
    1. Dropping duplicate rows.
    2. Removing known impossible outliers (e.g., age > 100, employment length > 100).
    """
    cleaned = df.copy()
    
    # 1. Drop duplicates
    initial_len = len(cleaned)
    cleaned = cleaned.drop_duplicates()
    
    # 2. Filter out impossible domain values discovered during EDA
    if "person_age" in cleaned.columns:
        cleaned = cleaned[cleaned["person_age"] <= 100]
        
    if "person_emp_length" in cleaned.columns:
        # Keep records where employment length is null (handled later by imputer) or <= 100
        cleaned = cleaned[(cleaned["person_emp_length"].isnull()) | (cleaned["person_emp_length"] <= 100)]
        
    cleaned = cleaned.reset_index(drop=True)
    return cleaned

def build_preprocessor() -> ColumnTransformer:
    """
    Constructs a ColumnTransformer that handles:
    - Missing values (Median imputation for numerical, Most Frequent for categorical)
    - Feature scaling (StandardScaler for numerical)
    - Categorical encoding (OneHotEncoder with handle_unknown='ignore')
    """
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, NUMERICAL_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )

    return preprocessor
