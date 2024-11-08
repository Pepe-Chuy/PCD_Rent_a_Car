import streamlit as st
import requests
import json
import pandas as pd
import numpy as np

df = pd.read_csv('/code/data/processed.csv')

cities = sorted(df['locationcity'].unique())
fuel = sorted(df['fuelType'].unique())
make = sorted(df['vehiclemake'].unique())
model = sorted(df['vehiclemodel'].unique())
type = sorted(df['vehicletype'].unique())
state = sorted(df['locationstate'].unique())

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
    Fueltype = st.selectbox("Select a fuel type", fuel)
    Rating = st.sidebar.slider("Rating", value=3.0, min_value=0.0, max_value=5.0)
    TripsTaken = st.sidebar.number_input("Trips taken", min_value=0, max_value=20, step=1)
    ReviewCount = st.sidebar.number_input("Review Count", min_value=0, max_value=1000, step=1)  # Added
    City = st.selectbox("Select a city", cities)
    Latitude = st.sidebar.number_input("Latitude", format="%.6f", min_value=-90.0, max_value=90.0, step=0.000001)
    Longitude = st.sidebar.number_input("Longitude", format="%.6f", min_value=-180.0, max_value=180.0, step=0.000001)    
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
        'reviewCount': ReviewCount,  # Added
        'location_city': City,  # Changed
        'location_latitude': Latitude,  # Changed
        'location_longitude': Longitude,  # Changed
        'location_state': State,  # Changed
        'owner_id': OwnerId,  # Changed
        'vehicle_make': VehicleMake,  # Changed
        'vehicle_model': VehicleModel,  # Changed
        'vehicle_type': VehicleType,  # Changed
        'vehicle_year': VehicleYear  # Changed
    }
    return input_dict


input_dict = user_input_features()
input_dict = convert_int64(input_dict)



if st.button('Predict'):
    try:
        response = requests.post(
            url="http://pcd-car-model-container:8000/predict",
            data=json.dumps(input_dict),
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            prediction = response.json().get('prediction', 'No prediction found')
            st.write(f"El precio estimado de la renta es: {prediction} dólares")
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# if st.button('Predict'):
#     response = requests.post(
#         #url="http://localhost:8000/predict",
#         url="http://pcd-car-model-container:8000/predict",
#         data=json.dumps(input_dict)
#     )

#     st.write(f"El precio estimado de la renta es: {response.json()['prediction']} dólares")