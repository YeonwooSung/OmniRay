import torch
from typing import Union
from enum import Enum

# custom modules
from .base import ServableInfo, DeviceType, Framework


class TorchModelType(Enum):
    TORCH_NN = "torch.nn"
    TORCH_SCRIPT = "torch.jit"


class TorchServableInfo(ServableInfo):
    def __init__(
        self,
        device_type: DeviceType = DeviceType.CPU,
        full_path: Union[str, None] = None,
        model_name: Union[str, None] = None,
        model_type: TorchModelType = TorchModelType.TORCH_NN,
        state_dict_path: Union[str, None] = None,
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

        else:
            raise ValueError(f"Unsupported device type {device_type} :: Torch model must be loaded on CUDA or CPU device.")

        self.model_type = model_type
        self.state_dict_path = state_dict_path
