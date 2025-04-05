import mlflow
import os
from mlflow.tracking import MlflowClient
from typing import Dict, Iterator, Optional, Any

# custom module
from .base import ModelRegistry


class MLflowModelRegistry(ModelRegistry):
    """
    MLflow implementation of the ModelRegistry interface.
    
    This class provides methods to register, retrieve, list, and manage models
    using MLflow's model registry.
    """

    def __init__(self, tracking_uri: Optional[str] = None, registry_uri: Optional[str] = None):
        """
        Initialize the MLflow model registry.
        
        Args:
            tracking_uri (str, optional): URI of the MLflow tracking server.
            registry_uri (str, optional): URI of the MLflow model registry.
        """
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        if registry_uri:
            mlflow.set_registry_uri(registry_uri)

        self.tracking_uri = tracking_uri
        self.registry_uri = registry_uri

        self.client = MlflowClient()
        self._models_cache: Dict[str, Any] = {}

    
    def register(self, model_file_path: str, model_name: str):
        """
        Register a model with the MLflow model registry.
        
        Args:
            model_file_path (str): Path to the model file.
            model_name (str): Name to register the model under.
            
        Returns:
            The registered model version.
        """
        if not os.path.exists(model_file_path):
            raise FileNotFoundError(f"Model file not found: {model_file_path}")
        
        # Check if model exists in registry
        try:
            self.client.get_registered_model(model_name)
        except mlflow.exceptions.MlflowException:
            # Create the model if it doesn't exist
            self.client.create_registered_model(model_name)
        
        # Log the model
        with mlflow.start_run():
            run_id = mlflow.active_run().info.run_id
            mlflow.log_artifact(model_file_path, "model")
            
            # Register the model
            model_uri = f"runs:/{run_id}/model"
            model_version = mlflow.register_model(model_uri, model_name)
            
        # Cache the model
        self._models_cache[model_name] = model_version
        
        return model_version
    
    def get(self, model_name: str):
        """
        Get a model from the registry by name.
        
        Args:
            model_name (str): Name of the model to retrieve.
            
        Returns:
            The latest version of the model.
            
        Raises:
            KeyError: If the model does not exist in the registry.
        """
        if model_name not in self:
            raise KeyError(f"Model '{model_name}' not found in registry")
        
        # Get the latest version of the model
        latest_version = self.client.get_latest_versions(model_name, stages=["None"])[0]
        model_uri = f"models:/{model_name}/{latest_version.version}"
        model = mlflow.pyfunc.load_model(model_uri)
        
        # Update cache
        self._models_cache[model_name] = model
        
        return model


    def list(self) -> list:
        """
        List all models in the registry.
        
        Returns:
            list: List of model names.
        """
        registered_models = self.client.list_registered_models()
        return [model.name for model in registered_models]


    def remove(self, model_name: str) -> None:
        """
        Remove a model from the registry.
        
        Args:
            model_name (str): Name of the model to remove.
            
        Raises:
            KeyError: If the model does not exist in the registry.
        """
        if model_name not in self:
            raise KeyError(f"Model '{model_name}' not found in registry")
        
        self.client.delete_registered_model(model_name)
        
        # Clean cache
        if model_name in self._models_cache:
            del self._models_cache[model_name]


    def clear(self) -> None:
        """
        Clear all models in the registry.
        """
        models = self.list()
        for model_name in models:
            self.remove(model_name)
        
        # Clear cache
        self._models_cache.clear()


    def __len__(self) -> int:
        """
        Return the number of models in the registry.
        
        Returns:
            int: Number of models in the registry.
        """
        return len(self.list())


    def __contains__(self, model_name) -> bool:
        """
        Check if a model exists in the registry.
        
        Args:
            model_name (str): Name of the model to check.
            
        Returns:
            bool: True if the model exists, False otherwise.
        """
        try:
            self.client.get_registered_model(model_name)
            return True
        except mlflow.exceptions.MlflowException:
            return False


    def __iter__(self) -> Iterator[str]:
        """
        Return an iterator over the model names in the registry.
        
        Returns:
            Iterator[str]: Iterator over model names.
        """
        return iter(self.list())


    def __getitem__(self, model_name):
        """
        Get a model from the registry by name.
        
        Args:
            model_name (str): Name of the model to retrieve.
            
        Returns:
            The model.
            
        Raises:
            KeyError: If the model does not exist in the registry.
        """
        return self.get(model_name)


    def __setitem__(self, model_name, model):
        """
        Add a model to the registry.
        
        Args:
            model_name (str): Name to register the model under.
            model: The model to register.
        """
        # Save the model to a temporary file
        import tempfile
        import joblib

        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as temp:
            temp_path = temp.name
            joblib.dump(model, temp_path)

        try:
            self.register(temp_path, model_name)
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_path):
                os.remove(temp_path)


    def __delitem__(self, model_name):
        """
        Remove a model from the registry.
        
        Args:
            model_name (str): Name of the model to remove.
            
        Raises:
            KeyError: If the model does not exist in the registry.
        """
        self.remove(model_name)
