from abc import ABC, abstractmethod

from .servable_info import ServableInfo


class BatchPredictor(ABC):
    def __init__(self, servable_info: ServableInfo):
        self.servable_info = servable_info

    @abstractmethod
    def predict(self, data):
        pass
