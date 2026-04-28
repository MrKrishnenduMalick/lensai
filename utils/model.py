"""model.py - Train and evaluate ML model."""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline


def preprocess_data(df: pd.DataFrame, target_col: str, sensitive_col: str):
    """
    Encode categorical columns, split features/target.
    Returns X, y, sensitive_series, feature_names, encoders dict.
    """
    df = df.copy().dropna()

    encoders = {}
    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    y = df[target_col]
    sensitive = df[sensitive_col]
    X = df.drop(columns=[target_col])

    feature_names = X.columns.tolist()
    return X, y, sensitive, feature_names, encoders


def train_model(X: pd.DataFrame, y: pd.Series):
    """
    Train LogisticRegression with standard scaling.
    Returns pipeline, X_test, y_test, y_pred.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42)),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred, output_dict=True),
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }

    return pipeline, y_pred, metrics
