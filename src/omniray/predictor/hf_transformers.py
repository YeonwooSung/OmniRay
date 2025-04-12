import ray


#TODO Implement Huggingface model inference
@ray.remote
class HuggingfacePredictor:
    def __init__(self, model_name: str):
        """
        Initialize the HuggingfacePredictor with a model name.

        Args:
            model_name (str): The name of the Hugging Face model to use.
        """
        self.model_name = model_name
        self.model = None
