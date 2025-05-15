import mlflow
from hyperopt import fmin, tpe, hp, Trials
import numpy as np

from data_prep import load_wine_data, prepare_data
from model import train_model

def objective(params, train_data):
    """
    Objective function for hyperparameter optimization.
    
    Args:
        params (dict): Hyperparameters to optimize
        train_data (tuple): Training data and related components
        
    Returns:
        dict: Results of model training
    """
    train_x, train_y, valid_x, valid_y, test_x, test_y, signature = train_data
    
    result = train_model(
        params=params,
        epochs=3,
        train_x=train_x,
        train_y=train_y,
        valid_x=valid_x,
        valid_y=valid_y,
        test_x=test_x,
        test_y=test_y,
        signature=signature
    )
    return result

def main():
    # Set the experiment name
    mlflow.set_experiment("/wine-quality")
    
    # Define hyperparameter search space
    space = {
        "lr": hp.loguniform("lr", np.log(1e-5), np.log(1e-1)),
        "momentum": hp.uniform("momentum", 0.0, 1.0),
    }
    
    # Load and prepare data
    data = load_wine_data()
    train_data = prepare_data(data)
    
    # Start MLflow run for hyperparameter optimization
    with mlflow.start_run():
        # Conduct hyperparameter search
        trials = Trials()
        best = fmin(
            fn=lambda params: objective(params, train_data),
            space=space,
            algo=tpe.suggest,
            max_evals=8,
            trials=trials,
        )
        
        # Get the best run
        best_run = sorted(trials.results, key=lambda x: x["loss"])[0]
        
        # Log the best parameters and metrics
        mlflow.log_params(best)
        mlflow.log_metric("best_eval_rmse", best_run["loss"])
        
        # Log the best model
        mlflow.tensorflow.log_model(best_run["model"], "model", signature=train_data[-1])
        
        # Print results
        print(f"Best parameters: {best}")
        print(f"Best eval RMSE: {best_run['loss']}")

if __name__ == "__main__":
    main()
