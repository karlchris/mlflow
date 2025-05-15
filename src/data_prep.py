import pandas as pd
from sklearn.model_selection import train_test_split
from mlflow.models import infer_signature

def load_wine_data(url="https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-white.csv"):
    """
    Load the wine quality dataset from the specified URL.
    
    Args:
        url (str): URL of the dataset
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    return pd.read_csv(url, sep=";")

def prepare_data(data, test_size=0.25, valid_size=0.2, random_state=42):
    """
    Prepare the data by splitting it into train, validation, and test sets.
    
    Args:
        data (pd.DataFrame): Input dataset
        test_size (float): Proportion of data to use for testing
        valid_size (float): Proportion of training data to use for validation
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_x, train_y, valid_x, valid_y, test_x, test_y, signature)
    """
    # Split into training and test sets
    train, test = train_test_split(data, test_size=test_size, random_state=random_state)
    
    # Prepare features and target variables
    train_x = train.drop(["quality"], axis=1).values
    train_y = train[["quality"]].values.ravel()
    test_x = test.drop(["quality"], axis=1).values
    test_y = test[["quality"]].values.ravel()
    
    # Split training data into train and validation sets
    train_x, valid_x, train_y, valid_y = train_test_split(
        train_x, train_y, test_size=valid_size, random_state=random_state
    )
    
    # Infer the model signature for MLflow
    signature = infer_signature(train_x, train_y)
    
    return train_x, train_y, valid_x, valid_y, test_x, test_y, signature

def get_feature_names():
    """
    Get the names of features in the wine quality dataset.
    
    Returns:
        list: List of feature names
    """
    return [
        "fixed acidity",
        "volatile acidity",
        "citric acid",
        "residual sugar",
        "chlorides",
        "free sulfur dioxide",
        "total sulfur dioxide",
        "density",
        "pH",
        "sulphates",
        "alcohol"
    ] 