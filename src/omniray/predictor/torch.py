import ray
import os
import torch
from typing import Union

# custom modules
from .base import Predictor
from .servable_info import TorchServableInfo, TorchModelType


@ray.remote
class TorchPredictor(Predictor):
    def __init__(
        self,
        servable_info: TorchServableInfo,
        model_obj: Union[torch.nn.Module, None] = None
    ):
        self.servable_info = servable_info

        self.model_name = servable_info.model_name
        self.device_type = servable_info.device_type

        if servable_info.model_type == TorchModelType.TORCH_NN:
            if model_obj is None:
                raise ValueError("model_cls must be provided for Torch nn.Module.")
            self.model = model_obj
            # load the model state dict if provided
            if servable_info.state_dict_path is not None and os.path.exists(servable_info.state_dict_path):
                self.model.load_state_dict(torch.load(servable_info.full_path))

        elif servable_info.model_type == TorchModelType.TORCH_SCRIPT:
            self.model = torch.jit.load(servable_info.full_path)

        else:
            raise ValueError(f"Unsupported model type {servable_info.model_type}")

        # move the model to the specified device
        _device = servable_info.device
        if _device is not None:
            self.model.to(_device)
        else:
            raise ValueError("Device type must be specified for Torch model.")


    @torch.inference_mode()
    def predict(self, data: dict):
        if self.servable_info.device is None:
            raise ValueError("Device type must be specified for Torch model.")

        inputs = {k: v.to(self.servable_info.device) for k, v in data.items()}
        outputs = self.model(**inputs)

        return outputs
