import pickle
import mlflow
from fastapi import FastAPI
from pydantic import BaseModel
from mlflow import MlflowClient


# MLflow settings
dagshub_repo = "https://dagshub.com/Pepe-Chuy/PCD_Rent_a_Car"

MLFLOW_TRACKING_URI = "https://dagshub.com/Pepe-Chuy/PCD_Rent_a_Car.mlflow"

mlflow.set_tracking_uri(uri=MLFLOW_TRACKING_URI)
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

run_ = mlflow.search_runs(order_by=['metrics.rmse ASC'],
                          output_format="list",
                          experiment_names=["Rent a Car"]
                          )[0]

run_id = run_.info.run_id

run_uri = f"runs:/{run_id}/preprocessor"

client.download_artifacts(
    run_id=run_id,
    path='preprocessor',
    dst_path='.'
)

with open("preprocessor/preprocessor.b", "rb") as f_in:
    dv = pickle.load(f_in)

model_name = "nyc-taxi-model"############################## PENDIENTE
alias = "champion"

model_uri = f"models:/{model_name}@{alias}"

champion_model = mlflow.pyfunc.load_model(
    model_uri=model_uri
)

def preprocess(input_data):

    input_dict = {
    'fuelType': input_data.fuelType,
    'rating': input_data.rating,
    'renterTripsTaken': input_data.renterTripsTaken,
    'location_city': input_data.location.city,
    'location_latitude': input_data.location.latitude,
    'location_longitude': input_data.location.longitude,
    'location_state': input_data.location.state,
    'owner_id': input_data.owner.id,
    'vehicle_make': input_data.vehicle.make,
    'vehicle_model': input_data.vehicle.model,
    'vehicle_type': input_data.vehicle.type,
    'vehicle_year': input_data.vehicle.year
}

    return dv.transform(input_dict)

def predict(input_data):

    X_pred = preprocess(input_data)

    return champion_model.predict(X_pred)


app = FastAPI()

class InputData(BaseModel):
    fuelType: int
    rating: float
    renterTripsTaken: int
    location_city: int
    location_latitude: float
    location_longitude: float
    location_state: int
    owner_id: int
    vehicle_make: int
    vehicle_model: int
    vehicle_type: int
    vehicle_year: int


@app.post("/predict")
def predict_endpoint(input_data: InputData):
    result = predict(input_data)[0]

    return {
        "prediction": float(result)
    }