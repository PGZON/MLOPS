import pandas as pd
import pickle
import os

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

from huggingface_hub import HfApi

# Load dataset
df = pd.read_csv("data/CleanedData.csv")

X = df[['name','company','year','kms_driven','fuel_type']]
y = df['Price']

categorical = ['name','company','fuel_type']
numeric = ['year','kms_driven']

# Preprocessing
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown="ignore"), categorical),
    ('num', 'passthrough', numeric)
])

# Pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

# Train model
pipeline.fit(X, y)

# Save model
os.makedirs("model", exist_ok=True)

model_path = "model/car_price_pipeline.pkl"

with open(model_path, "wb") as f:
    pickle.dump(pipeline, f)

print("Model saved locally")

# Read token
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN not found. Set it as environment variable.")

# Upload model
api = HfApi()

api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo="car_price_pipeline.pkl",
    repo_id="PG05/car-price-model",
    repo_type="model",
    token=HF_TOKEN
)

print("Model uploaded to HuggingFace")