import ray.data as rd
import datasets

# custom modules
from .base import BaseDataLoader


class HuggingfaceDataLoader(BaseDataLoader):
    def __init__(self):
        super().__init__()

    def load(self, path: str, **kwargs):
        self.dataset = datasets.load_dataset(path)
        split = kwargs.get("split", "train")

        try:
            use_all = kwargs.get("use_all", False)
            use_all = bool(use_all)
        except ValueError:
            use_all = False

        if use_all:
            self.ray_dataset = rd.from_huggingface(self.dataset)
        else:
            self.ray_dataset = rd.from_huggingface(self.dataset[split])
