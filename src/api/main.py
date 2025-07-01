from fastapi import FastAPI
from src.api.pydantic_models import CustomerData, PredictionResponse
import joblib
import pandas as pd
import mlflow 
import os

# Build absolute base directory
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Set correct model URI
model_uri = os.path.join(
    base_dir,
    "mlruns",
    "578058596288668658",
    "models",
    "m-c195880d3970492db5671ec190cb6a96",
    "artifacts"
)

# Optional but good practice
mlflow.set_tracking_uri("file://" + os.path.join(base_dir, "mlruns"))

# Load model
loaded_model = mlflow.sklearn.load_model(model_uri)

# Your custom feature creation function
from src.data_processing import TransactionFeatureExtractor, create_proxy_target

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Credit Risk Model API is up!"}

selected_columns = [
    "CustomerId",
    "ProviderId",
    "ProductId",
    "ProductCategory",
    "ChannelId",
    "Amount",
    "Value",
    "TransactionStartTime",
    "PricingStrategy",
    "FraudResult"  # Include this only if you're doing training
]

@app.post("/predict", response_model=PredictionResponse)
def predict(data: CustomerData):
    # Convert to DataFrame
    df = pd.DataFrame([data.dict()])
    print(df)
    filtered_df = df[selected_columns]
    
    # Create derived features (if you have such logic)
    df_new = TransactionFeatureExtractor().fit_transform(filtered_df)
    df_final = create_proxy_target(df_new)

    # Predict probability
    proba = loaded_model.predict_proba(df_final)[0][1]

    return PredictionResponse(risk_probability=proba)
