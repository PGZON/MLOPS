from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI()

# Load model
model = pickle.load(open("model/car_price_pipeline.pkl", "rb"))

# Request body schema
class CarFeatures(BaseModel):
    name: str
    company: str
    year: int
    kms_driven: int
    fuel_type: str


@app.get("/")
def health():
    return {"status": "model running"}


@app.post("/predict")
def predict(car: CarFeatures):

    # Convert request JSON → dataframe
    df = pd.DataFrame([car.dict()])

    prediction = model.predict(df)[0]

    return {"predicted_price": float(prediction)}