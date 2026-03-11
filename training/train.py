import pandas as pd
import pickle
import os

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

from huggingface_hub import HfApi

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

pipeline.fit(X,y)

os.makedirs("model", exist_ok=True)

model_path = "model/car_price_pipeline.pkl"

pickle.dump(pipeline, open(model_path,"wb"))

api = HfApi()

api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo="car_price_pipeline.pkl",
    repo_id="prathamesh0505/car-price-model",
    repo_type="model",
    token=os.environ["HF_TOKEN"]
)

print("Model uploaded to HuggingFace")