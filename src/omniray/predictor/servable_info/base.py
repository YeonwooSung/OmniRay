from enum import Enum
from typing import Union


class Framework(Enum):
    TORCH = "torch"
    TRANSFORMERS = "transformers"
    VLLM = "vllm"


class DeviceType(Enum):
    CPU = "cpu"
    CUDA = "cuda"
    METAL = "mps"


class ServableInfo:
    def __init__(
        self,
        framework: Framework = Framework.TORCH,
        device_type: DeviceType = DeviceType.CPU,
        full_path: Union[str, None] = None,
        model_name: Union[str, None] = None,
    ):
        self.framework = framework
        self.device_type = device_type
        self.full_path = full_path
        self.model_name = model_name
