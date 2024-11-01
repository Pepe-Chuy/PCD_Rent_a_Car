from prefect import flow, task
import pandas as pd
from sklearn.preprocessing import LabelEncoder


@task(name="Read Data", retries=5, retry_delay_seconds=20)
def read_data(filepath: str) -> pd.DataFrame:
    """Read raw data into DF"""
    data = pd.read_csv(filepath)
    return data


@task(name="Preprocess", retries=3, retry_delay_seconds=20)
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    "Encode the categorical variables"
    le = LabelEncoder()
    categorical_columns = df.select_dtypes(include=['object']).columns

    for col in categorical_columns:
        df[col] = le.fit_transform(df[col])

    return df
    


@task(name="Train-Test split", retries=5,retry_delay_seconds=20)
def train_test_split(df: pd.DataFrame, test_size: float) -> pd.DataFrame:
    "Split into train and test partitions"



@flow(name="MainFlow")
def mainFlow():
    data = read_data("")
    data = preprocess_data(data)
    x_train,x_test, y_train, y_test = train_test_split(data)
