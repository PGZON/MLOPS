from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle
import os

from huggingface_hub import hf_hub_download

app = FastAPI()

MODEL_PATH = hf_hub_download(
    repo_id="prathamesh0505/car-price-model",
    filename="car_price_pipeline.pkl",
    token=os.environ.get("HF_TOKEN")
)

model = pickle.load(open(MODEL_PATH, "rb"))


class CarFeatures(BaseModel):
    name: str
    company: str
    year: int
    kms_driven: int
    fuel_type: str


@app.get("/")
def health():
    return {"status": "model loaded"}


@app.post("/predict")
def predict(car: CarFeatures):

    df = pd.DataFrame([car.dict()])
    prediction = model.predict(df)[0]

    return {"predicted_price": float(prediction)}