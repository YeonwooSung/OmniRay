from abc import ABC, abstractmethod


class ModelRegistry(ABC):
    @abstractmethod
    def register(self, model_file_path: str, model_name: str):
        pass

    @abstractmethod
    def get(self, model_name: str):
        pass

    @abstractmethod
    def list(self) -> list:
        pass

    @abstractmethod
    def remove(self, model_name: str) -> None:
        """
        Remove a model from the registry.

        Args:
            model_name (str): Name of the model to remove.
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all models in the registry."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """
        Return the number of models in the registry.

        Returns:
            int: Number of models in the registry.
        """
        pass

    @abstractmethod
    def __contains__(self, model_name):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __getitem__(self, model_name):
        pass

    @abstractmethod
    def __setitem__(self, model_name, model):
        pass

    @abstractmethod
    def __delitem__(self, model_name):
        pass

    def __str__(self):
        class_name = self.__class__.__name__
        attr_str = ', '.join([f'{k}={v}' for k, v in self.__dict__.items()])
        return f'{class_name}({attr_str})'

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.__str__())
