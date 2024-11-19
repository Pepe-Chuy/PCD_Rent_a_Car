import streamlit as st
import requests
import json
import pandas as pd
import numpy as np

# Load data
raw = pd.read_csv('/code/data/raw.csv')

# Generate mapping dictionaries
mapping_dicts = {}
categorical_columns = ['locationcity', 'fuelType', 'vehiclemake', 'vehiclemodel', 'vehicletype', 'locationstate']
for column in categorical_columns:
    uniquevals = sorted(raw[column].unique())
    mapping_dicts[column] = {value: idx for idx, value in enumerate(uniquevals)}


# Streamlit app header
st.write("""
# Application to Predict the Price for Car Rental
""")

st.sidebar.header('User  Input Parameters')

def convert_int64(obj):
    """Helper function to convert np.int64 to int for JSON serialization."""
    if isinstance(obj, dict):
        return {key: convert_int64(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_int64(item) for item in obj]
    elif isinstance(obj, np.int64):
        return int(obj)
    else:
        return obj

def user_input_features():
    """Collect user inputs via Streamlit widgets and map them to backend-compatible values."""
    Fueltype = st.selectbox("Select a fuel type", list(mapping_dicts['fuelType'].keys()))
    Rating = st.sidebar.slider("Rating", value=3.0, min_value=0.0, max_value=5.0)
    TripsTaken = st.sidebar.number_input("Trips taken", min_value=0, max_value=20, step=1)
    ReviewCount = st.sidebar.number_input("Review Count", min_value=0, max_value=1000, step=1)
    City = st.selectbox("Select a city", list(mapping_dicts['locationcity'].keys()))
    Latitude = st.sidebar.number_input("Latitude", format="%.6f", min_value=-90.0, max_value=90.0, step=0.000001)
    Longitude = st.sidebar.number_input("Longitude", format="%.6f", min_value=-180.0, max_value=180.0, step=0.000001)
    State = st.selectbox("Select a state", list(mapping_dicts['locationstate'].keys()))
    OwnerId = st.sidebar.number_input("Owner ID", min_value=10000000, max_value=99999999, step=1)
    VehicleMake = st.selectbox("Select a Vehicle make", list(mapping_dicts['vehiclemake'].keys()))
    VehicleModel = st.selectbox("Select a Vehicle model", list(mapping_dicts['vehiclemodel'].keys()))
    VehicleType = st.selectbox("Select a Vehicle type", list(mapping_dicts['vehicletype'].keys()))
    VehicleYear = st.sidebar.slider("Vehicle Year", 1900, 2025, 2020)

    # Map user-friendly selections to integers using mapping_dicts
    input_dict = {
        'fuelType': mapping_dicts['fuelType'][Fueltype],
        'rating': Rating,
        'renterTripsTaken': TripsTaken,
        'reviewCount': ReviewCount,
        'location_city': mapping_dicts['locationcity'][City],
        'location_latitude': Latitude,
        'location_longitude': Longitude,
        'location_state': mapping_dicts['locationstate'][State],
        'owner_id': OwnerId,
        'vehicle_make': mapping_dicts['vehiclemake'][VehicleMake],
        'vehicle_model': mapping_dicts['vehiclemodel'][VehicleModel],
        'vehicle_type': mapping_dicts['vehicletype'][VehicleType],
        'vehicle_year': VehicleYear
    }
    return input_dict

# Collect user inputs
input_dict = user_input_features()
input_dict = convert_int64(input_dict)

# Predict button functionality
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