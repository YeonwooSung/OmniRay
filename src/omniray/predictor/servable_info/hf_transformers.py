import torch
from enum import Enum

# custom modules
from .base import ServableInfo, Framework, DeviceType


class HfModelType(Enum):
    TEXT = "text"
    SEQUENCE_CLASSIFICATION = "sequence-classification"
    TOKEN_CLASSIFICATION = "token-classification"


ENUM_OF_HF_MODELS = {
    HfModelType.TEXT.value,
    HfModelType.SEQUENCE_CLASSIFICATION.value,
    HfModelType.TOKEN_CLASSIFICATION.value,
}


class HfServableInfo(ServableInfo):
    def __init__(
        self,
        full_path: str,
        model_name: str,
        device_type: DeviceType = DeviceType.CUDA,
        max_batch_size: int = 1,
        type_of_model: str = "text",
    ):
        super().__init__(
            full_path=full_path,
            model_name=model_name,
            framework=Framework.TRANSFORMERS,
            device_type=device_type,
        )
        self.is_async_engine = False
        self.max_batch_size = max_batch_size

        if type_of_model not in ENUM_OF_HF_MODELS:
            raise ValueError(
                f"Unsupported HuggingFace model type {type_of_model}. Supported types are: {ENUM_OF_HF_MODELS}"
            )
        self.type_of_model = type_of_model

        if device_type not in {DeviceType.CUDA, DeviceType.CPU, DeviceType.METAL}:
            raise ValueError(f"Unsupported device type {device_type} :: HuggingFace model must be loaded on CUDA or CPU device.")

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
