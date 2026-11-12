import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERICAL_FEATURES = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]

CATEGORICAL_FEATURES = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]

TARGET_COLUMN = "loan_status"


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.drop_duplicates().copy()

    if "person_age" in cleaned.columns:
        cleaned = cleaned[cleaned["person_age"] <= 100]

    if "person_emp_length" in cleaned.columns:
        cleaned = cleaned[
            cleaned["person_emp_length"].isnull() | (cleaned["person_emp_length"] <= 100)
        ]

    return cleaned.reset_index(drop=True)


def build_preprocessor() -> ColumnTransformer:
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, NUMERICAL_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
