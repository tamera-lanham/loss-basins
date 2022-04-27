import json
import os
import shutil
from google.cloud import storage
from google.oauth2 import service_account
from pathlib import Path
from typing import Optional, Union


class GCS:
    def __init__(self, bucket_name: str, key_path: Optional[Union[Path, str]] = None):

        # Get the service account key
        if key_path is None:
            key_path = Path(os.getcwd(), "_keys", os.listdir("_keys")[0])
        key_path = Path(key_path)

        with open(key_path) as key_file:
            key = json.load(key_file)

        # Get the bucket
        storage_credentials = service_account.Credentials.from_service_account_info(key)
        client = storage.Client(
            project=key["project_id"], credentials=storage_credentials
        )
        self.bucket = client.get_bucket(bucket_name)

    def _upload_dir(self, local_path: Path, bucket_path: str):
        for path in local_path.iterdir():
            if path.is_file():
                self._upload_file(path, bucket_path + "/" + path.name)
            else:
                self._upload_dir(path, bucket_path + "/" + path.name)

    def _upload_file(self, local_path: Path, bucket_path: str):
        gcs_file = self.bucket.blob(bucket_path)
        gcs_file.upload_from_filename(local_path)

    def upload(self, local_path: Union[str, Path], bucket_path: str):
        local_path = Path(local_path)
        if local_path.is_dir():
            self._upload_dir(local_path, bucket_path)
        else:
            self._upload_file(local_path, bucket_path)
