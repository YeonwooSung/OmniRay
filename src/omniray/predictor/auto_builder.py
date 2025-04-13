from .base import Predictor
from .vllms import VllmPredictor
from .hf_transformers import HfModelPredictor, HfTextClassificationPredictor
from .servable_info import (
    ENUM_OF_HF_MODELS,
    ServableInfo,
    Framework,
    HfServableInfo,
)


class PredictorFactory:
    @staticmethod
    def from_servable_info(servable_info: ServableInfo) -> Predictor:
        if servable_info.framework == Framework.VLLM:
            return VllmPredictor(servable_info)

        if servable_info.framework == Framework.TRANSFORMERS:
            return PredictorFactory.get_hf_model_from_servable_info(servable_info)

        else:
            raise ValueError(f"Unsupported framework: {servable_info.framework}")

    @staticmethod
    def get_hf_model_from_servable_info(servable_info: HfServableInfo):
        if servable_info.type_of_model == "text":
            return HfModelPredictor(servable_info)
        elif servable_info.type_of_model == "sequence-classification":
            return HfTextClassificationPredictor(servable_info)
        elif servable_info.type_of_model == "token-classification":
            return HfTextClassificationPredictor(servable_info)
        else:
            raise ValueError(f"Unsupported HuggingFace model type {servable_info.type_of_model}. Supported types are: {ENUM_OF_HF_MODELS}")
