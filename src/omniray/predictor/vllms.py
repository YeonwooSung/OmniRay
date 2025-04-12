import ray

try:
    from vllm import AsyncLLMEngine, LLMEngine
except ImportError:
    from omniray.utils.logging import logger

    logger.log_warning(
        "VLLM is not installed. Please install it with `pip install vllm`."
        "VLLM models will not be available for use."
    )

# custom modules
from .base import Predictor
from .servable_info.vllm_servable import VllmConfigs, VllmServableInfo


@ray.remote
class VllmPredictor(Predictor):
    def __init__(self, servable_info: VllmServableInfo):
        super().__init__(servable_info)

        config: VllmConfigs = servable_info.config

        if servable_info.is_async_engine:
            self.engine = AsyncLLMEngine.from_engine_args(config.engine_args)
        else:
            self.engine = LLMEngine(config.engine_args)

    def is_async_engine(self):
        return self.servable_info.is_async_engine

    def predict(self, data):
        #TODO Implement VLLM model inference
        pass

    def predict_in_batch(self, data):
        #TODO Implement VLLM model inference
        pass
