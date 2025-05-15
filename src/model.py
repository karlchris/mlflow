import numpy as np
import keras
import mlflow
from hyperopt import STATUS_OK

def create_model(input_shape, mean, var):
    """
    Create a neural network model for wine quality prediction.
    
    Args:
        input_shape (tuple): Shape of input features
        mean (np.ndarray): Mean values for normalization
        var (np.ndarray): Variance values for normalization
        
    Returns:
        keras.Model: Compiled Keras model
    """
    model = keras.Sequential([
        keras.Input(input_shape),
        keras.layers.Normalization(mean=mean, variance=var),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dropout(0.1),
        keras.layers.Dense(1)
    ])
    return model

def compile_model(model, learning_rate, momentum):
    """
    Compile the model with specified optimizer settings.
    
    Args:
        model (keras.Model): The model to compile
        learning_rate (float): Learning rate for SGD optimizer
        momentum (float): Momentum for SGD optimizer
    """
    model.compile(
        optimizer=keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=momentum
        ),
        loss="mean_squared_error",
        metrics=[keras.metrics.RootMeanSquaredError()]
    )

def train_model(params, epochs, train_x, train_y, valid_x, valid_y, test_x, test_y, signature):
    """
    Train the model and log metrics with MLflow.
    
    Args:
        params (dict): Model hyperparameters
        epochs (int): Number of training epochs
        train_x (np.ndarray): Training features
        train_y (np.ndarray): Training targets
        valid_x (np.ndarray): Validation features
        valid_y (np.ndarray): Validation targets
        test_x (np.ndarray): Test features
        test_y (np.ndarray): Test targets
        signature (mlflow.models.ModelSignature): Model signature for MLflow
        
    Returns:
        dict: Training results including loss and model
    """
    # Calculate normalization statistics
    mean = np.mean(train_x, axis=0)
    var = np.var(train_x, axis=0)
    
    # Create and compile model
    model = create_model(train_x.shape[1:], mean, var)
    compile_model(model, params["lr"], params["momentum"])
    
    # Train model with MLflow tracking
    with mlflow.start_run(nested=True):
        # Train the model
        history = model.fit(
            train_x,
            train_y,
            validation_data=(valid_x, valid_y),
            epochs=epochs,
            batch_size=64,
            verbose=1
        )
        
        # Evaluate the model
        eval_result = model.evaluate(valid_x, valid_y, batch_size=64)
        eval_rmse = eval_result[1]
        
        # Log parameters and metrics
        mlflow.log_params(params)
        mlflow.log_metric("eval_rmse", eval_rmse)
        
        # Log training history
        for epoch, metrics in enumerate(history.history.items()):
            for metric_name, values in metrics:
                mlflow.log_metric(f"train_{metric_name}", values[epoch], step=epoch)
        
        # Log model
        mlflow.tensorflow.log_model(model, "model", signature=signature)
        
        return {
            "loss": eval_rmse,
            "status": STATUS_OK,
            "model": model,
            "history": history.history
        } 