import boto3
from botocore.client import Config
from boto3.s3.transfer import TransferConfig
import os
from typing import Union

from .base import ModelRegistry


class S3ModelRegistry(ModelRegistry):
    def __init__(
        self,
        bucket_name: str,
        prefix: Union[str, None] = None,
        max_pool_connections: int = 20,
        read_timeout: int = 900,
        transfer_max_concurrency: int = 20,
        transfer_use_threads: bool = True,
    ):
        self.bucket_name = bucket_name
        self.prefix = prefix
        self.s3_client_config = Config(
            max_pool_connections=max_pool_connections,
            read_timeout=read_timeout
        )
        self.s3 = boto3.client("s3", config=self.s3_client_config)
        self.transfer_config = TransferConfig(
            max_concurrency=transfer_max_concurrency,
            use_threads=transfer_use_threads,
        )


    def register(self, model_file_path: str, model_name: str):
        # Update model name with prefix
        if self.prefix is not None:
            model_name = os.path.join(self.prefix, model_name)

        # Upload model file to S3
        self.s3.upload_file(
            model_file_path,
            self.bucket_name,
            model_name,
            Config=self.transfer_config
        )


    def get(self, model_name: str):
        # Update model name with prefix
        if self.prefix is not None:
            model_name = os.path.join(self.prefix, model_name)

        # Download model file from S3
        self.s3.download_file(
            self.bucket_name,
            model_name,
            model_name,
            Config=self.transfer_config
        )


    def list(self) -> list:
        # List all objects in the bucket
        response = self.s3.list_objects_v2(Bucket=self.bucket_name)
        return [obj["Key"] for obj in response["Contents"]]


    def remove(self, model_name: str) -> None:
        # Update model name with prefix
        if self.prefix is not None:
            model_name = os.path.join(self.prefix, model_name)

        # Delete model file from S3
        self.s3.delete_object(Bucket=self.bucket_name, Key=model_name)


    def clear(self) -> None:
        # List all objects in the bucket
        response = self.s3.list_objects_v2(Bucket=self.bucket_name)

        # Delete all objects in the bucket
        for obj in response["Contents"]:
            self.s3.delete_object(Bucket=self.bucket_name, Key=obj["Key"])


    def __len__(self) -> int:
        model_list = self.list()
        return len(model_list)

    def __contains__(self, model_name):
        # use head_object to check if the object exists
        try:
            self.s3.head_object(Bucket=self.bucket_name, Key=model_name)
        except Exception:
            return False
        return True

    def __iter__(self):
        model_list = self.list()
        return iter(model_list)

    def __getitem__(self, model_name):
        self.get(model_name)

    def __setitem__(self, model_name, model):
        self.register(model, model_name)

    def __delitem__(self, model_name):
        self.remove(model_name)
