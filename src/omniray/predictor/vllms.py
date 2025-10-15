import ray
from typing import List, Dict, Any, Union, Optional
import asyncio

try:
    from vllm import AsyncLLMEngine, LLMEngine, SamplingParams
    from vllm.outputs import RequestOutput
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    from omniray.utils.logging import logger
    logger.log_warning(
        "VLLM is not installed. Please install it with `pip install vllm`. "
        "VLLM models will not be available for use."
    )

# custom modules
from .base import Predictor
from .servable_info.vllm_servable import VllmConfigs, VllmServableInfo


@ray.remote
class VllmPredictor(Predictor):
    """
    VLLM-based predictor for high-performance LLM inference.

    Supports both synchronous and asynchronous inference with batching capabilities.
    """

    def __init__(self, servable_info: VllmServableInfo):
        super().__init__(servable_info)

        if not VLLM_AVAILABLE:
            raise ImportError("VLLM is required but not installed. Install with: pip install vllm")

        config: VllmConfigs = servable_info.config

        if servable_info.is_async_engine:
            self.engine = AsyncLLMEngine.from_engine_args(config.engine_args)
        else:
            self.engine = LLMEngine.from_engine_args(config.engine_args)

        # Default sampling parameters
        self.default_sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=512
        )

    def is_async_engine(self):
        return self.servable_info.is_async_engine

    def predict(
        self,
        data: Union[str, Dict[str, Any]],
        sampling_params: Optional[SamplingParams] = None
    ) -> Union[str, RequestOutput]:
        """
        Perform inference on a single input.

        Args:
            data: Input text or dict containing 'prompt' key
            sampling_params: VLLM sampling parameters

        Returns:
            Generated text or RequestOutput object
        """
        if not VLLM_AVAILABLE:
            raise ImportError("VLLM is not available")

        # Extract prompt from data
        if isinstance(data, str):
            prompt = data
        elif isinstance(data, dict) and 'prompt' in data:
            prompt = data['prompt']
        else:
            raise ValueError("data must be a string or dict with 'prompt' key")

        # Use provided sampling params or default
        params = sampling_params or self.default_sampling_params

        if self.is_async_engine():
            # For async engine, use async predict
            return asyncio.run(self._async_predict_single(prompt, params))
        else:
            # Synchronous inference
            return self._sync_predict_single(prompt, params)

    def _sync_predict_single(
        self,
        prompt: str,
        sampling_params: SamplingParams
    ) -> str:
        """Synchronous single prediction."""
        request_id = f"request_{id(prompt)}"

        # Add request to engine
        self.engine.add_request(request_id, prompt, sampling_params)

        # Process until completion
        outputs = []
        while self.engine.has_unfinished_requests():
            step_outputs = self.engine.step()
            outputs.extend(step_outputs)

        # Extract generated text
        for output in outputs:
            if output.request_id == request_id:
                if output.outputs:
                    return output.outputs[0].text

        return ""

    async def _async_predict_single(
        self,
        prompt: str,
        sampling_params: SamplingParams
    ) -> str:
        """Asynchronous single prediction."""
        request_id = f"request_{id(prompt)}"

        # Generate with async engine
        results_generator = self.engine.generate(prompt, sampling_params, request_id)

        # Get final result
        final_output = None
        async for request_output in results_generator:
            final_output = request_output

        if final_output and final_output.outputs:
            return final_output.outputs[0].text

        return ""

    def predict_in_batch(
        self,
        data: List[Union[str, Dict[str, Any]]],
        sampling_params: Optional[SamplingParams] = None
    ) -> List[str]:
        """
        Perform batch inference on multiple inputs.

        Args:
            data: List of input texts or dicts containing 'prompt' key
            sampling_params: VLLM sampling parameters

        Returns:
            List of generated texts
        """
        if not VLLM_AVAILABLE:
            raise ImportError("VLLM is not available")

        # Extract prompts from data
        prompts = []
        for item in data:
            if isinstance(item, str):
                prompts.append(item)
            elif isinstance(item, dict) and 'prompt' in item:
                prompts.append(item['prompt'])
            else:
                raise ValueError("Each item must be a string or dict with 'prompt' key")

        # Use provided sampling params or default
        params = sampling_params or self.default_sampling_params

        if self.is_async_engine():
            # For async engine, use async batch predict
            return asyncio.run(self._async_predict_batch(prompts, params))
        else:
            # Synchronous batch inference
            return self._sync_predict_batch(prompts, params)

    def _sync_predict_batch(
        self,
        prompts: List[str],
        sampling_params: SamplingParams
    ) -> List[str]:
        """Synchronous batch prediction."""
        # Add all requests to engine
        request_ids = []
        for i, prompt in enumerate(prompts):
            request_id = f"batch_request_{i}"
            request_ids.append(request_id)
            self.engine.add_request(request_id, prompt, sampling_params)

        # Process all requests
        outputs_dict = {rid: None for rid in request_ids}

        while self.engine.has_unfinished_requests():
            step_outputs = self.engine.step()

            for output in step_outputs:
                if output.finished:
                    outputs_dict[output.request_id] = output

        # Extract generated texts in order
        results = []
        for request_id in request_ids:
            output = outputs_dict.get(request_id)
            if output and output.outputs:
                results.append(output.outputs[0].text)
            else:
                results.append("")

        return results

    async def _async_predict_batch(
        self,
        prompts: List[str],
        sampling_params: SamplingParams
    ) -> List[str]:
        """Asynchronous batch prediction."""
        # Create tasks for all prompts
        tasks = []
        for i, prompt in enumerate(prompts):
            request_id = f"async_batch_request_{i}"
            task = self._async_generate_single(prompt, sampling_params, request_id)
            tasks.append(task)

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        return results

    async def _async_generate_single(
        self,
        prompt: str,
        sampling_params: SamplingParams,
        request_id: str
    ) -> str:
        """Helper for async generation of a single prompt."""
        results_generator = self.engine.generate(prompt, sampling_params, request_id)

        final_output = None
        async for request_output in results_generator:
            final_output = request_output

        if final_output and final_output.outputs:
            return final_output.outputs[0].text

        return ""

    def set_default_sampling_params(
        self,
        temperature: float = 0.8,
        top_p: float = 0.95,
        top_k: int = -1,
        max_tokens: int = 512,
        **kwargs
    ):
        """
        Update default sampling parameters.

        Args:
            temperature: Sampling temperature
            top_p: Top-p (nucleus) sampling
            top_k: Top-k sampling
            max_tokens: Maximum tokens to generate
            **kwargs: Additional sampling parameters
        """
        self.default_sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_tokens,
            **kwargs
        )
