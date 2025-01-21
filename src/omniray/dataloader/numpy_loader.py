import numpy as np
import ray.data as rd

# custom modules
from .base import BaseDataLoader


class NumpyDataLoader(BaseDataLoader):
    def __init__(self):
        super().__init__()

    def load(self, path: str, **kwargs):
        self.dataset = np.load(path)
        self.ray_dataset = rd.from_numpy(self.dataset)

    def load_from_ndarray(self, ndarray: np.ndarray):
        self.dataset = ndarray
        self.ray_dataset = rd.from_numpy(self.dataset)
