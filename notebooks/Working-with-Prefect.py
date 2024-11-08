from prefect import flow, task
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import sklearn.metrics as metrics
from sklearn.model_selection import GridSearchCV
import mlflow
from mlflow import MlflowClient

@task(name="Read Data", retries=5, retry_delay_seconds=20)
def read_data(filepath: str) -> pd.DataFrame:
    """Read raw data into DF"""
    data = pd.read_csv(filepath)
    return data

# @task(name="Preprocess", retries=3, retry_delay_seconds=20)
# def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
#     "Encode the categorical variables"
#     le = LabelEncoder()
#     categorical_columns = df.select_dtypes(include=['object']).columns

#     for col in categorical_columns:
#         df[col] = le.fit_transform(df[col])

#     return df
    
@task(name="Train-Test split", retries=5,retry_delay_seconds=20)
def tt_split(df: pd.DataFrame, test_size: float) -> tuple:
    "Split into train and test partitions"

    X = df.drop(columns=["ratedaily"]) 
    Y = df["ratedaily"]

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)

    return X_train, X_test, y_train, y_test


@task(name="Scaling", retries = 5, retry_delay_seconds = 15)
def scaler( X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

@task(name="Train Model", retries=5, retry_delay_seconds=15)
def train_model(X_train_scaled, y_train, X_test_scaled, y_test):
    modelo = xgb.XGBRegressor(objective='reg:squarederror', seed=42)
    modelo.fit(X_train_scaled, y_train)
    y_hat = modelo.predict(X_test_scaled)
    
    r2 = metrics.r2_score(y_test, y_hat)
    mse = metrics.mean_squared_error(y_test, y_hat)
    
    # Log metrics to MLflow
    mlflow.log_metric("R2", r2)
    mlflow.log_metric("MSE", mse)
    print('R2:', r2)
    print("MSE:", mse)

    # Log the model
    mlflow.xgboost.log_model(modelo, "model")  # Log the model
    print("Model logged successfully.")
    return r2, mse

@task(name = "Grid Search", retries =5, retry_delay_seconds = 15)
def best_par(df: pd.DataFrame, X_train_scaled, y_train):
    gbm_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10, 15], 
    'learning_rate': [0.001, 0.01, 0.05]  
    }

    gbm = xgb.XGBRegressor(objective='reg:squarederror', seed=42)

    grid_mse = GridSearchCV(
        estimator=gbm,
        param_grid=gbm_param_grid,
        scoring='neg_mean_squared_error', 
        cv=10,  
        verbose=1  
    )

    grid_mse.fit(X_train_scaled, y_train)

    best_params = grid_mse.best_params_
    return best_params

@task(name= "Best model", retries =5, retry_delay_seconds = 15)
def best_model(df: pd.DataFrame, best_params, X_train, y_train):

    modelo_nuevo = xgb.XGBRegressor(
    objective='reg:squarederror',
    seed=42,
    **best_params  
    )

    modelo_nuevo.fit(X_train, y_train)
    return modelo_nuevo


# @task(name= "Predict",retries =5, retry_delay_seconds = 15)
# def predict(df: pd.DataFrame, modelo_nuevo, X_test, y_test):
#     y_hat= modelo_nuevo.predict(X_test)

#     r2 = metrics.r2_score(y_test,y_hat)
#     mse = metrics.mean_squared_error(y_test,y_hat)
#     print('R2:',r2)
#     print("MSE:",mse)
#     return print(("R2", r2), ("MSE", mse))


@task(name = 'Register Model')
def register_model():
    MLFLOW_TRACKING_URI = mlflow.get_tracking_uri()
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    df = mlflow.search_runs(order_by=['metrics.MSE'])  # Note the uppercase 'MSE'
    print(df.head())  # Inspect the DataFrame to ensure it has the expected columns
    if df.empty:
        raise ValueError("No runs found. Please ensure that models are logged correctly.")
    
    run_id = df.loc[df['metrics.MSE'].idxmin()]['run_id']
    run_uri = f"runs:/{run_id}/model"

    result = mlflow.register_model(
        model_uri=run_uri,
        name="rental-cars-perfect"
    )
    model_name = "beer-money-perfect"
    model_version_alias = "champion"

    # Create "champion" alias for version 1 of model "nyc-taxi-model"
    client.set_registered_model_alias(
        name=model_name,
        alias=model_version_alias,
        version='1'
    )

@flow(name="MainFlow")
def mainFlow(filepath: str, test_size: float = 0.2):
    data = read_data(filepath)
    # data = preprocess_data(data)  # Uncomment if you want to preprocess the data
    X_train, X_test, y_train, y_test = tt_split(data, test_size=test_size)
    X_train_scaled, X_test_scaled = scaler(X_train, X_test)  # Corrected line
    best_params = best_par(data, X_train_scaled, y_train)
    best_model_trained = best_model(data, best_params, X_train_scaled, y_train)
    register_model()


if __name__ == "__main__":
    mainFlow(filepath="../data/processed.csv")