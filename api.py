from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load model + pipeline
model = joblib.load("model.pkl")
pipeline = joblib.load("pipeline.pkl")

@app.get("/")
def home():
    return {"message": "Housing Price Prediction API"}

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    transformed = pipeline.transform(df)
    prediction = model.predict(transformed)
    return {"prediction": float(prediction[0])}

#--------------------------------------------------------------
#code 2
# @app.get("/")
# def home():
#     return {"message": "Housing API"}

# @app.post("/predict")
# def predict(data: dict):
#     result = predict_single(data)
#     return {"prediction": result}
#--------------------------------------------------------------