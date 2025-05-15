# MLflow Wine Quality Prediction

This repository demonstrates a complete MLflow project for predicting wine quality using machine learning. It includes experiment tracking, model management, and deployment capabilities.

## Features

- Machine Learning pipeline for wine quality prediction
- Experiment tracking with MLflow
- Hyperparameter optimization using Hyperopt
- Model serving capabilities
- Containerized deployment with Docker
- Comprehensive logging and metrics tracking

## Project Structure

```
.
├── src/
│   ├── main.py           # Main training script
│   ├── data_prep.py      # Data preparation utilities
│   └── model.py          # Model architecture definition
├── notebooks/            # Jupyter notebooks for exploration
├── tests/               # Unit tests
├── config/              # Configuration files
├── Dockerfile           # Container definition
├── Makefile            # Build and deployment commands
└── requirements.txt     # Python dependencies
```

## Prerequisites

- Python 3.12+
- Docker
- Make

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd mlflow
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Local Development

1. Run the training script:
```bash
python src/main.py
```

2. View MLflow UI:
```bash
mlflow ui
```

### Docker Deployment

1. Build the container:
```bash
make build
```

2. Start the MLflow server:
```bash
make server
```

3. Stop the server:
```bash
make stop
```

## Model Training

The project uses a neural network to predict wine quality based on physicochemical properties. The model architecture includes:
- Input normalization
- Dense layers with ReLU activation
- Hyperparameter optimization for learning rate and momentum

## MLflow Tracking

The following metrics and parameters are tracked:
- Learning rate
- Momentum
- RMSE on validation set
- Model architecture
- Training parameters

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 