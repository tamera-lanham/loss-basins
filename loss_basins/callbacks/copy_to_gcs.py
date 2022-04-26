import os
from pathlib import Path
from typing import Optional, Union
from pytorch_lightning.callbacks import Callback
import torch as t

from loss_basins.utils import GCS


class CopyToGCS(Callback):
    def __init__(
        self,
        output_path: Union[Path, str],
        gcs_bucket: str,
        gcs_path: str,
        gcs_key_path: Optional[Union[Path, str]] = None,
    ):
        self.output_path = Path(output_path)
        self.gcs_bucket = gcs_bucket
        self.gcs_path = gcs_path

        self.gcs = GCS(bucket_name=gcs_bucket, key_path=gcs_key_path)

    def on_fit_end(self, _, __):
        self.gcs.upload(self.output_path, self.gcs_path)
