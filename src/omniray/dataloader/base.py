import ray
from ray.data.dataset import MaterializedDataset
from typing import Callable, Union
import torch


class BaseDataLoader:
    def __init__(self, init_ray: bool = False):
        if init_ray:
            ray.init(ignore_reinit_error=True)

        self.ray_dataset: Union[MaterializedDataset, None] = None

    def load(self, path: str, **kwargs):
        raise NotImplementedError("load method is not implemented.")


    def get_ray_dataset(self) -> MaterializedDataset:
        if self.ray_dataset is None:
            raise ValueError("ray_dataset is None. Please load the dataset first.")
        return self.ray_dataset


    def lambda_map_chained(self, funcs: list):
        if isinstance(self.ray_dataset, MaterializedDataset):
            for func in funcs:
                self.ray_dataset = self.ray_dataset.map(func)
            return self.ray_dataset
        raise TypeError(f"The type of ray_dataset is unknown: {type(self.ray_dataset)}")


    def lambda_map(self, func: Callable) -> MaterializedDataset:
        if self.ray_dataset is None:
            raise ValueError("ray_dataset is None. Please load the dataset first.")

        if isinstance(self.ray_dataset, MaterializedDataset):
            return self.ray_dataset.map(func)
        raise TypeError(f"The type of ray_dataset is unknown: {type(self.ray_dataset)}")


    def lambda_map_batch(
        self,
        func: Callable,
        concurrency: int = 2,
        batch_size: int = 1,
        num_cpu: int = 1
    ) -> MaterializedDataset:
        """
        This method is used to apply a function to each batch of the dataset.

        Args:
            func (Callable): The function to apply to each batch.
            concurrency (int): The number of concurrent tasks to use.
            batch_size (int): The number of elements to include in each batch.
            num_cpu (int): The number of CPUs to allocate per ray worker.

        Returns:
            MaterializedDataset: The new dataset with the function applied to each batch.
        """
        if self.ray_dataset is None:
            raise ValueError("ray_dataset is None. Please load the dataset first.")

        if isinstance(self.ray_dataset, MaterializedDataset):
            # Reference: <https://docs.ray.io/en/latest/data/batch_inference.html#batch-inference-home>
            return self.ray_dataset.map_batches(
                func,
                num_cpus=num_cpu,
                batch_size=batch_size,
                concurrency=concurrency,
            )
        raise TypeError(f"The type of ray_dataset is unknown: {type(self.ray_dataset)}")


    def lambda_map_batch_gpu(
        self,
        func: Callable,
        concurrency: int = 1,
        batch_size: int = 1,
        num_gpu: int = 1
    ) -> MaterializedDataset:
        """
        This method is used to apply a function to each batch of the dataset.

        Args:
            func (Callable): The function to apply to each batch.
            concurrency (int): The number of concurrent tasks to use.
            batch_size (int): The number of elements to include in each batch.
            num_gpu (int): The number of GPUs to allocate per ray worker.

        Returns:
            MaterializedDataset: The new dataset with the function applied to each batch.
        """
        if self.ray_dataset is None:
            raise ValueError("ray_dataset is None. Please load the dataset first.")

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, cannot allocate GPU resources.")

        if isinstance(self.ray_dataset, MaterializedDataset):
            # Reference: <https://docs.ray.io/en/latest/data/batch_inference.html#batch-inference-home>
            return self.ray_dataset.map_batches(
                func,
                num_gpus=num_gpu,
                batch_size=batch_size,
                concurrency=concurrency,
            )
        raise TypeError(f"The type of ray_dataset is unknown: {type(self.ray_dataset)}")
