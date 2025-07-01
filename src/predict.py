import pandas as pd
import mlflow.sklearn
import os, sys

sys.path.append(os.path.abspath('..'))  # adds the parent directory

# Point directly to the model folder under artifacts
model_uri = r"file:///d:/Project/10 Academy Resourse/credit risk/credit-risk-model/mlruns/578058596288668658/models/m-c195880d3970492db5671ec190cb6a96/artifacts/"

loaded_model = mlflow.sklearn.load_model(model_uri)

df = pd.read_csv('notebooks\\testfile.csv')

pred = loaded_model.predict(df)
print(pred)
# Use the loaded model
# Example: predictions = loaded_model.predict(X_test)
print("✅ Model loaded successfully:", loaded_model)
