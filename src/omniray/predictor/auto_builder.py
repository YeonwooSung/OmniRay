from .base import BatchPredictor
from .vllms import VllmPredictor

from .servable_info import ServableInfo, Framework


class BatchPredictorFactory:
    @staticmethod
    def from_servable_info(servable_info: ServableInfo) -> BatchPredictor:
        if servable_info.framework == Framework.VLLM:
            return VllmPredictor(servable_info)
        else:
            raise ValueError(f"Unsupported framework: {servable_info.framework}")
