import torch
import ray.data as rd

# custom modules
from .base import BaseDataLoader


class TorchDataLoader(BaseDataLoader):
    def __init__(self):
        super().__init__()

    def load(self, path: str, **kwargs):
        self.dataset = torch.load(path)
        self.ray_dataset = rd.from_torch(self.dataset)
