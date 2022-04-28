import json
import os
from pathlib import Path
from typing import Optional, Union
from pytorch_lightning.callbacks import Callback
import torch as t


class SaveModelState(Callback):
    def __init__(
        self,
        output_path: Union[Path, str],
        every_n_epochs: int = 1,
        every_n_batches: int = 0,
        save_init: bool = True,
        save_val_outputs: bool = True,
    ):
        self.output_path = Path(output_path)
        self.every_n_epochs = every_n_epochs
        self.every_n_batches = every_n_batches
        self.save_init = save_init
        self.save_val_outputs = save_val_outputs

        self.metrics_file = self.output_path / "metrics.jsonl"

        # Create dir for parameter checkpoints
        self.parameter_checkpoints_dir = self.output_path / "parameter_checkpoints"
        if not self.parameter_checkpoints_dir.exists():
            os.makedirs(self.parameter_checkpoints_dir)

        if self.save_val_outputs:
            # Create dir for val outputs
            self.val_outputs_dir = self.output_path / "val_ouputs"
            if not self.val_outputs_dir.exists():
                os.makedirs(self.val_outputs_dir)

            self.val_outputs = []

    def _save_state(self, trainer, pl_module, epoch=None, batch=None):
        state_name = self._state_name(epoch, batch)

        # Save parameters
        t.save(pl_module.state_dict(), self.parameter_checkpoints_dir / state_name)

        # Save metrics
        metrics = {
            "epoch": epoch,
            "batch": "end" if batch is None else batch,
            **{k: v.item() for k, v in trainer.callback_metrics.items()},
        }
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(metrics) + "\n")

    def _state_name(self, epoch, batch=None):
        return f"epoch_{epoch}" + (f"_batch_{batch}" if batch is not None else "_end")

    def on_fit_start(self, trainer, pl_module):
        if self.save_init:
            self._save_state(trainer, pl_module, epoch=-1)

    def on_train_batch_end(self, trainer, pl_module, _, __, batch_idx):
        if (
            self.every_n_batches
            and batch_idx != 0
            and batch_idx % self.every_n_batches == 0
        ):
            self._save_state(
                trainer, pl_module, epoch=trainer.current_epoch, batch=batch_idx
            )

    def on_epoch_end(self, trainer, pl_module):
        if self.every_n_epochs and trainer.current_epoch % self.every_n_epochs == 0:
            self._save_state(trainer, pl_module, epoch=trainer.current_epoch)

    def on_validation_batch_end(self, _, __, outputs, *___):
        if self.save_val_outputs:
            self.val_outputs.append(outputs)

    def on_validation_epoch_end(self, trainer, _) -> None:
        if self.save_val_outputs:
            print(self.val_outputs)

    def on_train_epoch_end(self, trainer, pl_module):
        if self.every_n_epochs and trainer.current_epoch % self.every_n_epochs == 0:
            self._save_state(trainer, pl_module, epoch=trainer.current_epoch)
