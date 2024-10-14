import mlflow

# custom module
from .base import ModelRegistry

#TODO implement the MLflow registry

class MLflowModelRegistry(ModelRegistry):
    def __init__(self, tracking_uri: str, experiment_name: str):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name

        # Set MLflow tracking URI
        mlflow.set_tracking_uri(tracking_uri)

        # Set MLflow experiment
        mlflow.set_experiment(experiment_name)

    def register(self, model_file_path: str, model_name: str):
        pass

    def get(self, model_name: str):
        pass

    def list(self) -> list:
        pass

    def remove(self, model_name: str) -> None:
        pass

    def clear(self) -> None:
        pass

    def __len__(self) -> int:
        pass

    def __contains__(self, model_name):
        pass

    def __iter__(self):
        pass

    def __getitem__(self, model_name):
        pass

    def __setitem__(self, model_name, model):
        pass

    def __delitem__(self, model_name):
        self.remove(model_name)
