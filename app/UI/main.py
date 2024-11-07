import streamlit as st
import requests
import json
import pandas as pd
import numpy as np

df = pd.read_csv('/code/data/processed.csv')

cities = sorted(df['location.city'].unique())
fuel = sorted(df['fuelType'].unique())
make = sorted(df['vehicle.make'].unique())
model = sorted(df['vehicle.model'].unique())
type = sorted(df['vehicle.type'].unique())
state = sorted(df['location.state'].unique())

st.write("""
# Application to predict the price for car rental
""")

st.sidebar.header('User Input Parameters')

def convert_int64(obj):
    if isinstance(obj, dict):
        return {key: convert_int64(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_int64(item) for item in obj]
    elif isinstance(obj, np.int64):
        return int(obj)
    else:
        return obj


def user_input_features():
    Fueltype = st.selectbox("Select a fueltypr", fuel)
    Rating = st.sidebar.slider("Rating", value = 3, min_value= 0, max_value=5)
    TripsTaken = st.sidebar.number_input("Trips taken", min_value=0, max_value=20, step=1)
    City = st.selectbox("Select a city", cities)
    Latitude = st.sidebar.number_input("Latitude 12.345678", format="%.6f", min_value=-90.0, max_value=90.0, step=0.000001)
    Longitude = st.sidebar.number_input("Longitude -12.345678", format="%.6f", min_value=-180.0, max_value=180.0, step=0.000001)    
    State = st.selectbox("Select a state", state)
    OwnerId = st.sidebar.number_input("Owner ID", min_value=10000000, max_value=99999999, step=1)
    VehicleMake = st.selectbox("Select a Vehicle make", make)
    VehicleModel = st.selectbox("Select a Vehicle model", model)
    VehicleType = st.selectbox("Select a Vehicle type", type)
    VehicleYear = st.sidebar.slider("Vehicle Year", 1900, 2025, 2020)
            
    input_dict = {
        'fuelType': Fueltype,
        'rating': Rating,
        'renterTripsTaken': TripsTaken,
        'location_city': City,
        'location_latitude': Latitude,
        'location_longitude': Longitude,
        'location_state': State,
        'owner_id': OwnerId,
        'vehicle_make': VehicleMake,
        'vehicle_model': VehicleModel,
        'vehicle_type': VehicleType,
        'vehicle_year': VehicleYear
    }

    return input_dict

input_dict = user_input_features()

input_dict = convert_int64(input_dict)

if st.button('Predict'):
    response = requests.post(
        #url="http://localhost:8000/predict",
        url="http://pcd-car-model-container:8000/predict",
        data=json.dumps(input_dict)
    )

    st.write(f"El precio estimado de la renta es: {response.json()['prediction']} dólares")