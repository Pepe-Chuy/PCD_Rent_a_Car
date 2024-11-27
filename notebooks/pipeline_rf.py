from prefect import flow, task
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor as rfr
import numpy as np
import mlflow
from sklearn import metrics
from hyperopt import fmin, tpe, Trials, hp
import dagshub

# DagsHub initialization
dagshub.init(url="https://dagshub.com/Pepe-Chuy/PCD_Rent_a_Car", mlflow=True)
MLFLOW_TRACKING_URI = mlflow.get_tracking_uri()
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(experiment_name="Rent a Car")

@task(name="Read Data", retries=5, retry_delay_seconds=20)
def read_data(filepath: str) -> pd.DataFrame:
    """Read raw data into DataFrame"""
    data = pd.read_csv(filepath)
    return data

@task(name="Train-Test Split", retries=5, retry_delay_seconds=20)
def tt_split(df: pd.DataFrame, test_size: float) -> tuple:
    """Split into train and test partitions"""
    X = df.drop(columns=["ratedaily"])
    y = df["ratedaily"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    return X_train, X_test, y_train, y_test

@task(name="Train Best Model", retries=2)
def train_best_model(X_train, y_train):
    """Train model using hyperparameter optimization"""
    def objective(params):
        with mlflow.start_run(nested=True):
            # Model family
            mlflow.set_tag("model_family", "RandomForest")

            # Log parameters
            mlflow.log_params(params)

            # Train model
            model = rfr(
                n_estimators=int(params['n_estimators']),
                max_depth=int(params['max_depth']),
                min_samples_split=int(params['min_samples_split']),
                min_samples_leaf=int(params['min_samples_leaf']),
                random_state=309
            )
            model.fit(X_train, y_train)

            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
            rmse = np.sqrt(-cv_scores.mean())  # Convert negative MSE to RMSE

            # Log metrics
            mlflow.log_metric("RMSE", rmse)
            return rmse

    # Hyperparameter search space
    search_space = {
        'n_estimators': hp.quniform('n_estimators', 10, 20, 1),
        'max_depth': hp.quniform('max_depth', 5, 10, 1),
        'min_samples_split': hp.quniform('min_samples_split', 2, 10, 1),
        'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 10, 1),
    }

    trials = Trials()

    # Hyperparameter optimization
    with mlflow.start_run(run_name="Father Random Forest Regressor", nested=True):
        best_params = fmin(
            fn=objective,
            space=search_space,
            algo=tpe.suggest,
            max_evals=10,
            trials=trials
        )

    return best_params

@task(name="Log Best Model", retries=2)
def log_best_model(X_train, X_test, y_train, y_test, best_params):
    """Log the best model and its metrics"""
    with mlflow.start_run(run_name="Best Random Forest Model"):
        # Train the best model
        best_model = rfr(
            n_estimators=int(best_params['n_estimators']),
            max_depth=int(best_params['max_depth']),
            min_samples_split=int(best_params['min_samples_split']),
            min_samples_leaf=int(best_params['min_samples_leaf']),
            random_state=309
        )
        best_model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = best_model.predict(X_test)
        rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        r_squared = metrics.r2_score(y_test, y_pred)

        # Log parameters and metrics
        mlflow.log_params(best_params)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R_squared", r_squared)

@flow(name="Random Forest Training Flow")
def random_forest_pipeline(filepath: str, test_size: float):
    df = read_data(filepath)
    X_train, X_test, y_train, y_test = tt_split(df, test_size)
    best_params = train_best_model(X_train, y_train)
    log_best_model(X_train, X_test, y_train, y_test, best_params)


random_forest_pipeline(filepath="../data/processed.csv", test_size=0.2)
