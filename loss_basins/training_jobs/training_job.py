from concurrent.futures import ProcessPoolExecutor as Pool
from multiprocessing import get_context

from dataclasses import dataclass, asdict
import json
import os
from pathlib import Path
import pytorch_lightning as pl
import subprocess
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
    n_processes: int = 4

    def __post_init__(self):
        self.most_recent_commit_hash = self._current_commit_hash()

    def _current_commit_hash(self) -> str:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )


class TrainingJob:
    def __init__(self, metadata: Metadata):
        self.metadata = metadata
        self.output_path = Path("./_data/", self.__class__.__name__ + "_" + timestamp())

        self.default_trainer_kwargs = {
            "accelerator": "gpu" if t.cuda.is_available() else "cpu",
            "logger": False,
            "num_processes": 1,
            "enable_checkpointing": False,
        }

    def model(self, metadata: Metadata) -> pl.LightningModule:
        raise NotImplementedError("Should be implemented by subclass")

    def data_loaders(
        self, metadata: Metadata
    ) -> Tuple[t.utils.data.DataLoader, t.utils.data.DataLoader]:
        raise NotImplementedError("Should be implemented by subclass")

    def callbacks(self, init_id: str, init_metadata: Metadata) -> Iterable[pl.Callback]:
        progress_bar = pl.callbacks.TQDMProgressBar()

        checkpoint = SaveModelState(
            self.output_path / "inits" / init_id, every_n_epochs=1
        )

        return [progress_bar, checkpoint]

    def trainer(self, init_metadata: Metadata) -> pl.Trainer:
        raise NotImplementedError("Should be implemented by subclass")

    def run_init(self, init_metadata: Metadata, trainer: pl.Trainer):

        train_loader, val_loader = self.data_loaders(init_metadata)
        model = self.model(init_metadata)

        trainer.fit(model, train_loader, val_loader)

    def run(self):
        self._save_training_job_info()

        init_metadata_iter = self.generate_init_metadata(self.metadata)
        with Pool(self.metadata.n_processes, mp_context=get_context("spawn")) as pool:
            pool.map(self._run_init_wrapper, init_metadata_iter)

        if self.metadata.gcs_bucket:
            self._save_outputs_to_gcs()

    def generate_init_metadata(
        self, training_job_metadata: Metadata
    ) -> Iterable[Tuple[str, Metadata]]:

        for i in range(training_job_metadata.n_init_repeats):
            init_id, init_metadata = str(i), training_job_metadata
            yield init_id, init_metadata

    def _run_init_wrapper(self, init_id_and_metadata: Tuple[str, Metadata]):
        init_id, init_metadata = init_id_and_metadata

        self._save_init_metadata(init_id, init_metadata)
        trainer = self.trainer(init_metadata)
        trainer.callbacks += self.callbacks(init_id, init_metadata)

        # May have to manually set trainer.gpus here from process rank if pl doesn't schedule properly
        gpu_index = init_id % self.metadata.n_processes
        if t.cuda.is_available():
            trainer.gpus = [gpu_index]

        self.run_init(init_metadata, trainer)

    def _test_run(self):
        # Just to see if something isn't working
        self.run_init(self.metadata, self.trainer(self.metadata))

    def _save_training_job_info(self):
        if not self.output_path.exists():
            os.makedirs(self.output_path)

        # Save training job metadata
        with open(self.output_path / "training_job_metadata.json", "w") as f:
            json.dump(asdict(self.metadata), f)

        # Save the model as torchscript
        # Model initialization is currently using job metadata, not init metadata
        model = self.model(self.metadata)
        scripted_model = model.to_torchscript()
        t.jit.save(scripted_model, self.output_path / "model_torchscript.pt")

    def _save_init_metadata(self, init_id: str, init_metadata: Metadata):
        with open(self.output_path / "init_metadata.jsonl", "a") as f:
            metadata_dict = {"init_id": init_id, **asdict(init_metadata)}
            f.write(json.dumps(metadata_dict) + "\n")

    def _save_outputs_to_gcs(self):
        print(
            f"Saving outputs from ./{self.output_path} to gcs://{self.metadata.gcs_bucket}/{self.output_path.name}/ ..."
        )
        gcs = GCS(self.metadata.gcs_bucket)
        gcs.upload(self.output_path, self.output_path.name)
        print("Done.")
