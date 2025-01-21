import pandas as pd
import ray.data as rd

# custom modules
from .base import BaseDataLoader


class PandasDataLoader(BaseDataLoader):
    def __init__(self):
        super().__init__()

    def load(self, path: str, **kwargs):
        if path.endswith(".csv"):
            self.dataset = pd.read_csv(path)
        elif path.endswith(".parquet"):
            self.dataset = pd.read_parquet(path)
        elif path.endswith(".xlsx"):
            self.dataset = pd.read_excel(path)
        else:
            raise ValueError(f"Unsupported file format: {path}")

        self.ray_dataset = rd.from_pandas(self.dataset)


    def load_from_dataframe(self, dataframe: pd.DataFrame):
        self.dataset = dataframe
        self.ray_dataset = rd.from_pandas(self.dataset)
