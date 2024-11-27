---

# Rent a Car - Data Science Project

## Introduction

In this data science project, we aim to develop a model to estimate the daily rental price of a car based on its specific characteristics. By applying techniques learned in class, we will identify patterns and relationships among various vehicle attributes, such as manufacturing year, brand, model, fuel type, mileage, and more, to provide precise rental price predictions.

Additionally, we will utilize **FastAPI** to develop a web application that allows users to interact with the system in a simple and fast way. This application is essential in a market where car rentals are increasingly popular. An accurate tool will benefit both car owners, by setting competitive prices, and customers, by offering fair rates based on data.

---

## Background

The model will be trained using a car rental dataset from Kaggle. This process involves data cleaning, feature selection, and model evaluation. The ultimate goal is to provide a valuable tool for vehicle owners to set competitive prices while offering customers fair, data-driven rates.

---

## Objectives

### General Objective

Develop a data science solution to predict the daily rental price of a car, applying machine learning and MLOps techniques. This will optimize decision-making in car rental companies by implementing a predictive model deployed as a web API using **FastAPI**.

### Specific Objectives

- Conduct exploratory data analysis (EDA) on the car rental dataset to identify patterns and relationships among key variables (e.g., manufacturing year, brand, model, fuel type, mileage).
- Preprocess the data, including cleaning, transformation, and feature selection, to ensure quality.
- Train and validate various machine learning models to select the one that best predicts daily rental prices.
- Develop an API using **FastAPI** for users to interact with the predictive model.
- Deploy the API in the cloud to ensure accessibility and scalability, enabling real-time rental price queries.

---

## Problem Statement

In the car rental market, both owners and customers face challenges related to pricing. The lack of precise tools for estimating daily rental prices based on vehicle characteristics results in inefficient pricing and unsatisfactory user experiences.

Our solution is to develop a predictive model and deploy it through a web API using **FastAPI**. This will enable precise price estimation, optimizing pricing strategies and enhancing transparency in the market.

---

## Modeling

We conducted several experiments to evaluate which approach provided the best results. We utilized models like **RandomForestRegressor** with **GridSearch** to select and log the best-performing model. Initially orchestrated using a notebook, the workflow was later implemented in **MLflow**. The selected model was **RandomForest**.

---

## Pipeline

The **Prefect** pipeline includes the following tasks:

- **Read Data:** Reads the path and converts it into a DataFrame.
- **Train-Test-Split:** Splits the data into training and testing sets, separating features (X) and target (Y).
- **Train Best Model:** Trains the RandomForest model using GridSearch.
- **Log Best Model:** Logs the best model in MLflow.

---

## Application

The application consists of two containers orchestrated using a `docker-compose.yaml` file. It includes:

1. **Backend:** Built with **FastAPI**, it loads the best model from MLflow for predictions.
2. **Frontend:** Developed using **Streamlit**, providing a user-friendly interface for inputting vehicle details to predict rental prices.

Files for backend and frontend include:
- `main.py`
- `Dockerfile`
- `requirements.txt`

---

## Deployment

The application was deployed on an **AWS EC2 instance**, hosting both backend and frontend images. This setup allows users to connect and make API requests via the EC2 instance, ensuring efficient and functional operations.

---

## Conclusions

This project provided a comprehensive understanding of the entire lifecycle of a machine learning model, from model logging and versioning to creating a workflow with Prefect. Furthermore, we successfully developed a functional application with backend and frontend, deploying it on the cloud using an EC2 instance.

While challenging, the learnings from this project will be invaluable for future endeavors, enabling us to turn models into tangible, functional tools.

---