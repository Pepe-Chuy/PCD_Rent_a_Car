import pickle
import mlflow
from fastapi import FastAPI
from pydantic import BaseModel
from mlflow import MlflowClient
import pandas as pd
import mlflow


dagshub_repo = "https://dagshub.com/Pepe-Chuy/PCD_Rent_a_Car"

MLFLOW_TRACKING_URI = "https://dagshub.com/Pepe-Chuy/PCD_Rent_a_Car.mlflow"

mlflow.set_tracking_uri(uri=MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


logged_model = 'runs:/fa7792b4494c4baab57fe253f8c321ca/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)


data = {
    "fuelType": 1,
    "rating": 5.0,
    "renterTripsTaken": 13,
    "reviewCount": 12,
    "locationcity": 722,
    "locationlatitude": 47.449107,
    "locationlongitude": -122.308841,
    "locationstate": 43,
    "ownerid": 12847615,
    "rate.daily": 135,
    "vehiclemake": 48,
    "vehiclemodel": 288,
    "vehicletype": 2,
    "vehicleyear": 2019
}

 
input_data = pd.DataFrame([data])

# Predict using the loaded model
print(loaded_model.predict(input_data))