# scripts/train.py
import pandas as pd
import mlflow
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from .data_processing import TransactionFeatureExtractor


def train_model(model_name, numeric_cols, categorical_cols, df_data):
    print(f"🔄 Training with model: {model_name}")
    
    df = df_data.copy()
    df.drop(columns=['TransactionId', 'BatchId', 'AccountId', 'SubscriptionId'], inplace=True)
    
    y = df['is_high_risk']
    X = df.drop(columns=['is_high_risk'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "DecisionTree": DecisionTreeClassifier(),
        "RandomForest": RandomForestClassifier(),
        "GradientBoosting": GradientBoostingClassifier()
    }

    if model_name not in models:
        raise ValueError(f"❌ Model {model_name} not supported. Choose from: {list(models.keys())}")

    model = models[model_name]

    pipeline = Pipeline([
        ('feature_engineer', TransactionFeatureExtractor()),
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    print("Available columns:", X_train.columns.tolist())
    input('waiting for user input to continue...')

    print("🚀 Fitting pipeline...")
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)

    with mlflow.start_run(run_name=model_name):
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_dict(classification_report(y_test, y_pred, output_dict=True), "metrics.json")
        mlflow.sklearn.log_model(pipeline, model_name)

    print(f"✅ Model: {model_name}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("🎯 Model logged to MLflow.")
