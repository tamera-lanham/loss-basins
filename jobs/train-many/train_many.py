from ast import Param
from dataclasses import dataclass, field, asdict
import datetime
from click import progressbar
from google.cloud import storage
from google.oauth2 import service_account
import json
import os
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import TQDMProgressBar
import random
import string
import torch as t
from typing import Callable, Optional
from loss_basins.data import mnist_loader
from loss_basins.models.lightning import LightningModel, SaveModelState
from loss_basins.models.mnist_conv import MnistConv


def train_loader_fn():
    return mnist_loader(100, train=True, num_workers=8)


def val_loader_fn():
    return mnist_loader(100, train=False, num_workers=1)


def build_params():
    n_gpus = t.cuda.device_count()
    use_gpu = bool(n_gpus)

    params = Parameters(
        n_models=1000,
        n_processes=n_gpus if use_gpu else 2,
        device_type="gpu" if use_gpu else "cpu",
        model_fn=MnistConv,
        train_loader_fn=train_loader_fn,
        val_loader_fn=val_loader_fn,
        optimizer_kwargs={"lr": 1e-3},
        batch_size=100,
        loss_threshold=2.2,
    )

    return params


########################################################################


@dataclass
class Parameters:
    n_models: int
    n_processes: int
    device_type: str

    model_fn: Callable[[], t.nn.Module]
    train_loader_fn: Callable[[], t.utils.data.DataLoader]
    val_loader_fn: Callable[[], t.utils.data.DataLoader]

    optimizer_class: type = t.optim.SGD
    optimizer_kwargs: dict = field(default_factory=lambda: {"lr": 1e-3})
    loss_fn: Callable = t.nn.functional.cross_entropy

    batch_size: int = 100
    loss_threshold: float = 0.25

    output_dir: Optional[Path] = None
    save_to_gcs: bool = True
    gcs_bucket_name: str = "loss-basins"
    gcs_output_dir: Optional[str] = None

    def __post_init__(self):
        model_class = self.model_fn().__class__.__name__
        default_output_dirname = f"{model_class}_x{self.n_models}_{timestamp()}"

        if self.output_dir is None:
            self.output_dir = Path(f"_data/{default_output_dirname}")

        if self.save_to_gcs and self.gcs_output_dir is None:
            self.gcs_output_dir = default_output_dirname


def random_id(n=8):
    char_set = string.ascii_uppercase + string.ascii_lowercase + string.digits
    return "".join(random.choice(char_set) for _ in range(n))


def timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d--%H-%M")


def setup(params: Parameters):

    if not params.output_dir.exists():
        os.makedirs(params.output_dir)

    t.save(asdict(params), params.output_dir.joinpath("params.pt"))

    # TODO: Trace and save torchscript

    if params.save_to_gcs:
        bucket = get_gcs_bucket(params.gcs_bucket_name)
        gcs_file = bucket.blob(params.gcs_output_dir + "/params.pt")
        gcs_file.upload_from_filename(params.output_dir.joinpath("params.pt"))


def get_gcs_bucket(name="loss-basins"):
    with open("_keys/trim-keel-346317-aae7b66dbc69.json") as source:
        info = json.load(source)

    storage_credentials = service_account.Credentials.from_service_account_info(info)
    client = storage.Client(project=info["project_id"], credentials=storage_credentials)
    bucket = client.get_bucket(name)
    return bucket


def save_dir_to_gcs(directory: Path, bucket, gcs_dir: str):
    for filename in os.listdir(directory):
        filepath = directory.joinpath(filename)
        gcs_path = "/".join(gcs_dir.split("/") + [directory.name, filepath.name])
        gcs_file = bucket.blob(gcs_path)
        gcs_file.upload_from_filename(filepath)


def train_one(rank, params: Parameters, output_path):

    if params.device_type == "gpu":
        device_n = rank
    else:
        device_n = -1

    model = LightningModel(
        params.model_fn(),
        params.loss_fn,
        params.optimizer_class,
        params.optimizer_kwargs,
    )

    train_loader, val_loader = params.train_loader_fn(), params.val_loader_fn()

    progress_bar = TQDMProgressBar(process_position=rank)
    checkpoint = SaveModelState(output_path, every_n_epochs=1, every_n_batches=300)
    early_stop = EarlyStopping("val_loss", stopping_threshold=params.loss_threshold)

    trainer = pl.Trainer(
        accelerator=params.device_type,
        devices=device_n,
        num_processes=1,
        callbacks=[progress_bar, checkpoint, early_stop],
        enable_checkpointing=False,
    )

    trainer.fit(model, train_loader, val_loader)

    return model, trainer


# def train_and_save_one(params: Parameters):

#     if params.save_to_gcs:
#         gcs_bucket = get_gcs_bucket(params.gcs_bucket_name)

#     output_path = params.output_dir.joinpath(random_id())

#     if not params.output_dir.exists():
#         os.makedirs(params.output_dir)

#     train_one(params, output_path)

#     if params.save_to_gcs:
#         save_dir_to_gcs(output_path, gcs_bucket, params.gcs_output_dir)


def train_many(rank, params: Parameters):
    process_n_models = params.n_models // params.n_processes
    if params.n_models % params.n_processes > rank:
        process_n_models += 1

    print(f"Process {rank} training {process_n_models} models")

    if params.save_to_gcs:
        gcs_bucket = get_gcs_bucket(params.gcs_bucket_name)

    for _ in range(process_n_models):
        output_path = params.output_dir.joinpath(random_id())

        if not params.output_dir.exists():
            os.makedirs(params.output_dir)

        train_one(rank, params, output_path)

        if params.save_to_gcs:
            save_dir_to_gcs(output_path, gcs_bucket, params.gcs_output_dir)


def init_processes(rank, params, fn, backend="gloo"):
    """Initialize the distributed environment."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    t.distributed.init_process_group(backend, rank=rank, world_size=params.n_processes)
    fn(rank, params)


def main():
    params = build_params()
    setup(params)

    processes = []
    t.multiprocessing.set_start_method("spawn", force=True)
    for rank in range(params.n_processes):
        p = t.multiprocessing.Process(
            target=init_processes, args=(rank, params, train_many)
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


if __name__ == "__main__":

    # params = build_params()
    # model, trainer = train_one(params)

    main()
