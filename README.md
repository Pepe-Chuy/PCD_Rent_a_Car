# Car Rental Reservation System

## Overview

This repository contains a web application for managing car rentals. The system allows users to rent cars from a dealership, view available cars, make reservations, and track rental history. It includes features for managing user profiles, car inventory, and reservation details.

## Features

- *User Management:* Register and manage user profiles, including contact information and driver’s license details.
- *Car Inventory:* View and manage a list of cars available for rent, including details such as make, model, year, and color.
- *Reservation System:* Create, update, and track car rental reservations with start and end dates.
- *Dynamic Pricing:* (Optional) Implement dynamic pricing based on demand and availability.
- *Fraud Detection:* (Optional) Monitor and prevent fraudulent activities.
- *Recommendation System:* (Optional) Recommend cars based on user preferences and past behavior.

## Data

The repository includes sample CSV files for demonstration purposes:

- users.csv: Contains information about users who have rented cars, including their contact details and registration dates.
- cars_rentals.csv: Contains information about the cars, their rental periods, and the users who rented them.

## Technologies

- *Backend:* [Framework/Language] (FastAPI, Python, Docker)
- *Frontend:* [Framework/Language] (Streamlit Docker)

## Setup

## notebooks



## app
implementation of front and backend of an app built on a 2 docker images, connected and setted up by a docker compose, the backend is built on python, and fastapi, it picks a model from the model registry in mlflow and uses it for the prediction, the frontend is settted up in streamlit, it displays all necesary variables for the user to fill 