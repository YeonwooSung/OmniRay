import ray

try:
    from vllm import AsyncLLMEngine, LLMEngine, RequestOutput, SamplingParams
except ImportError:
    from omniray.utils.logging import logger

    logger.log_warning(
        "VLLM is not installed. Please install it with `pip install vllm`."
        "VLLM models will not be available for use."
    )

from .base import BatchPredictor
from .servable_info.vllm_servable_info import VLLMServableInfo, VLLMConfigs


@ray.remote
class VLLMBatchPredictor(BatchPredictor):
    def __init__(self, servable_info: VLLMServableInfo):
        super().__init__(servable_info)

        config: VLLMConfigs = servable_info.config

        if servable_info.is_async_engine:
            self.engine = AsyncLLMEngine.from_engine_args(config.engine_args)
        else:
            self.engine = LLMEngine(config.engine_args)

    def is_async_engine(self):
        return self.servable_info.is_async_engine

    def predict(self, data):
        #TODO Implement VLLM model inference
        pass
