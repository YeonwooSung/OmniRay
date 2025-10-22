# OmniRay

Ray for all ML tasks.

![OmniRay Architecture](./assets/imgs/architecture.png)

Note: This repository is still in active development, and is not stable for production.

## ToDo

- Data Loader
    - [x] Add support for huggingface datasets
    - [x] Add support for PyTorch DataLoader
    - [x] Add support for numpy DataLoader
    - [x] Add support for pandas DataLoader
    - [x] Add support for Kafka streaming DataLoader
- Model Registry
    - [x] Add support for S3 registry
    - [x] Add support for MLFlow registry
- Batch Predictor
    - [x] Add support for vLLM batch predictor
    - [x] Add support for PyTorch batch predictor
    - [x] Add support for HuggingFace (with safetensor) batch predictor
- [ ] Preprocessor
- [ ] Postprocessor

## References

- [Scaling Pinterest ML Infrastructure with Ray: From Training to End-to-End ML Pipelines](https://medium.com/pinterest-engineering/scaling-pinterest-ml-infrastructure-with-ray-from-training-to-end-to-end-ml-pipelines-4038b9e837a0)
- [Ray Batch Inference at Pinterest (Part 3)](https://medium.com/pinterest-engineering/ray-batch-inference-at-pinterest-part-3-4faeb652e385)
- [Batch Predictions in Ray](https://docs.ray.io/en/latest/ray-core/examples/batch_prediction.html)
