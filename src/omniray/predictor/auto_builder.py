from .base import Predictor
from .vllms import VllmPredictor

from .servable_info import ServableInfo, Framework


class PredictorFactory:
    @staticmethod
    def from_servable_info(servable_info: ServableInfo) -> Predictor:
        if servable_info.framework == Framework.VLLM:
            return VllmPredictor(servable_info)
        else:
            raise ValueError(f"Unsupported framework: {servable_info.framework}")
