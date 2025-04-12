import torch

# custom modules
from .base import ServableInfo, Framework, DeviceType


class TransformersServableInfo(ServableInfo):
    def __init__(
        self,
        full_path: str,
        model_name: str,
        device_type: DeviceType = DeviceType.CUDA,
    ):
        super().__init__(
            full_path=full_path,
            model_name=model_name,
            framework=Framework.TRANSFORMERS,
            device_type=device_type,
        )
        self.is_async_engine = False

        if device_type not in {DeviceType.CUDA, DeviceType.CPU, DeviceType.METAL}:
            raise ValueError(f"Unsupported device type {device_type} :: HuggingFace model must be loaded on CUDA or CPU device.")

        self.device_type = device_type

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
