import pandas as pd
import pickle
import os

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

print("Loading dataset...")

df = pd.read_csv("data/CleanedData.csv")

X = df[['name','company','year','kms_driven','fuel_type']]
y = df['Price']

categorical = ['name','company','fuel_type']
numeric = ['year','kms_driven']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown="ignore"), categorical),
    ('num', 'passthrough', numeric)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

print("Training model...")

pipeline.fit(X, y)

os.makedirs("model", exist_ok=True)

model_path = "model/car_price_pipeline.pkl"

print("Saving model...")

with open(model_path, "wb") as f:
    pickle.dump(pipeline, f)

print("Model saved at:", model_path)