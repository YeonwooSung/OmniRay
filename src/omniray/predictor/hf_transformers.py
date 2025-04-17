import ray
import torch
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

# custom modules
from .base import Predictor
from .servable_info import HfServableInfo, ENUM_OF_HF_MODELS, HfModelType


@ray.remote
class HfModelPredictor(Predictor):
    def __init__(self, servable_info: HfServableInfo):
        """
        Initialize the HuggingfacePredictor with a model name.

        Args:
            model_name (str): The name of the Hugging Face model to use.
        """
        self.servable_info = servable_info

        self.model_name = servable_info.model_name
        self.model = AutoModel.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model.eval()

        if servable_info.device is not None:
            self.model.to(servable_info.device)


    def predict(self, data):
        if isinstance(data, str):
            inputs = self.tokenizer(data, return_tensors="pt")
        elif isinstance(data, list):
            inputs = self.tokenizer(data, return_tensors="pt", padding=True)
        elif isinstance(data, torch.Tensor):
            inputs = data
        else:
            raise ValueError("Unsupported input type. Expected str, list, or torch.Tensor.")

        if self.servable_info.device is not None:
            inputs = {k: v.to(self.servable_info.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs


@ray.remote
class HfTextClassificationPredictor(Predictor):
    def __init__(self, servable_info: HfServableInfo):
        super().__init__(servable_info)

        self.model_name = servable_info.model_name
        self.device = servable_info.device

        if servable_info.type_of_model == HfModelType.SEQUENCE_CLASSIFICATION.value:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        elif servable_info.type_of_model == HfModelType.TOKEN_CLASSIFICATION.value:
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)
        else:
            raise ValueError(f"Unsupported HuggingFace model type {servable_info.type_of_model}. Supported types are: {ENUM_OF_HF_MODELS}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model.eval()
        if servable_info.device is not None:
            self.model.to(servable_info.device)


    def predict(self, data):
        if isinstance(data, str):
            inputs = self.tokenizer(data, return_tensors="pt")
        elif isinstance(data, list):
            inputs = self.tokenizer(data, return_tensors="pt", padding=True)
        elif isinstance(data, torch.Tensor):
            inputs = data
        else:
            raise ValueError("Unsupported input type. Expected str, list, or torch.Tensor.")

        if self.device is not None:
            if isinstance(inputs, torch.Tensor):
                inputs = inputs.to(self.device)
                with torch.no_grad():
                    outputs = self.model(inputs)
            else:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.model(**inputs)

        return outputs
