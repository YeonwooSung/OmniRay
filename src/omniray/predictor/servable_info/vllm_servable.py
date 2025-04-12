try:
    from vllm import EngineArgs, SamplingParams, AsyncEngineArgs
except ImportError:
    from omniray.utils.logging import logger

    logger.log_warning(
        "vllm is not installed. Please install it with `pip install vllm`."
        "vllm models will not be available for use."
    )
from typing import Union

from .base import ServableInfo, Framework, DeviceType


class VllmConfigs:
    engine_args: Union[EngineArgs, AsyncEngineArgs]
    sampling_params: SamplingParams


class VllmServableInfo(ServableInfo):
    def __init__(
        self,
        full_path: str,
        model_name: str,
        device_type: DeviceType = DeviceType.CUDA,
        config: Union[VllmConfigs, None] = None
    ):
        super().__init__(
            full_path=full_path,
            model_name=model_name,
            framework=Framework.VLLM,
            device_type=device_type,
        )

        # check if VLLMConfigs is provided
        assert config is not None, "VLLMConfigs must be provided."

        if device_type is not DeviceType.CUDA:
            if device_type is DeviceType.CPU:
                print("vllm model is loaded on CPU device.")
                print("This may cause a significant slowdown in model inference.")
            else:
                raise ValueError("vllm model must be loaded on CUDA device.")

        self.config = config

        if not isinstance(self.config, VllmConfigs):
            raise ValueError("config must be an instance of VLLMConfigs.")

        # check if engine_args is either EngineArgs or AsyncEngineArgs, and set is_async_engine
        if isinstance(self.config.engine_args, EngineArgs):
            self.is_async_engine = False
        elif isinstance(self.config.engine_args, AsyncEngineArgs):
            self.is_async_engine = True
        else:
            raise ValueError("engine_args must be an instance of EngineArgs or AsyncEngineArgs.")
