import torch
from typing import Union

# custom modules
from .base import ServableInfo, DeviceType, Framework


class TorchServableInfo(ServableInfo):
    def __init__(
        self,
        device_type: DeviceType = DeviceType.CPU,
        full_path: Union[str, None] = None,
        model_name: Union[str, None] = None
    ):
        super().__init__(Framework.TORCH, device_type, full_path, model_name)

        if device_type not in {DeviceType.CUDA, DeviceType.CPU, DeviceType.METAL}:
            raise ValueError(f"Unsupported device type {device_type} :: Torch model must be loaded on CUDA or CPU device.")

        if device_type is DeviceType.CUDA:
            if not torch.cuda.is_available():
                raise ValueError("CUDA is not available. Please check your CUDA installation.")
            self.device = torch.device("cuda")

        elif device_type is DeviceType.CPU:
            self.device = torch.device("cpu")

        elif device_type is DeviceType.METAL:
            if not torch.backends.mps.is_available():
                raise ValueError("MPS is not available. Please check your MPS installation.")
            self.device = torch.device("mps")
