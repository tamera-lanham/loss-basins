from concurrent.futures import ProcessPoolExecutor as Pool
from dataclasses import dataclass, asdict
import json
import os
from pathlib import Path
import pytorch_lightning as pl
import torch as t
from typing import Iterable, Optional, Tuple

from loss_basins.callbacks import SaveModelState
from loss_basins.utils import timestamp, GCS


@dataclass
class Metadata:
    job_type: str = "TrainingJob"
    description: str = ""
    most_recent_commit_hash: str = ""
    n_init_repeats: int = 1
    gcs_bucket: Optional[str] = None


class TrainingJob:
    def __init__(self, metadata: Metadata):
        self.metadata = metadata
        self.output_path = Path("./_data/", self.__class__.__name__ + "_" + timestamp())
        if not self.output_path.exists():
            os.makedirs(self.output_path)

        self._save_training_job_metadata()
        self._save_model()

    def _save_training_job_metadata(self):
        with open(self.output_path / "training_job_metadata.json", "w") as f:
            json.dump(asdict(self.metadata), f)

    def _save_model(self):
        # Currently using job metadata, not init metadata
        model = self.model(self.metadata)

        scripted_model = model.to_torchscript()
        t.jit.save(scripted_model, self.output_path / "model_torchscript.pt")

    def model(self, metadata: Metadata):
        raise NotImplementedError()

    def data_loaders(self, metadata: Metadata):
        raise NotImplementedError()

    def callbacks(self, init_id: str, init_metadata: Metadata):
        progress_bar = pl.callbacks.TQDMProgressBar()

        checkpoint = SaveModelState(
            self.output_path / "inits" / init_id, every_n_epochs=1
        )

        return [progress_bar, checkpoint]

    def trainer(self, init_metadata: Metadata):
        raise NotImplementedError()

    def generate_init_metadata(
        self, training_job_metadata: Metadata
    ) -> Iterable[Tuple[str, Metadata]]:

        for i in range(training_job_metadata.n_init_repeats):
            init_id, init_metadata = str(i), training_job_metadata
            yield init_id, init_metadata

    def run_init(self, init_metadata: Metadata, trainer: pl.Trainer):
        raise NotImplementedError()

    def _save_init_metadata(self, init_id: str, init_metadata: Metadata):
        with open(self.output_path / "init_metadata.jsonl", "a") as f:
            metadata_dict = {"init_id": init_id, **asdict(init_metadata)}
            f.write(json.dumps(metadata_dict) + "\n")

    def _run_init_wrapper(self, init_id_and_metadata: Tuple[str, Metadata]):
        init_id, init_metadata = init_id_and_metadata

        self._save_init_metadata(init_id, init_metadata)
        trainer = self.trainer(init_metadata)
        trainer.callbacks = trainer.callbacks + self.callbacks(init_id, init_metadata)

        self.run_init(init_metadata, trainer)

    def run_dist(self, n_processes=4):
        init_metadata_iter = self.generate_init_metadata(self.metadata)
        with Pool(n_processes) as pool:
            pool.map(self._run_init_wrapper, init_metadata_iter)

        if self.metadata.gcs_bucket:
            self._save_outputs_to_gcs()

    def _save_outputs_to_gcs(self):
        print(
            f"Saving outputs from ./{self.output_path} to gcs://{self.metadata.gcs_bucket}/{self.output_path.name}/ ..."
        )
        gcs = GCS(self.metadata.gcs_bucket)
        gcs.upload(self.output_path, self.output_path.name)
        print("Done.")