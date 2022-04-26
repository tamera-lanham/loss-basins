from google.cloud import storage
from google.oauth2 import service_account
import json
import os
from pathlib import Path
import datetime

def timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d--%H-%M")

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


if __name__=="__main__":
    bucket = get_gcs_bucket()
    directory = Path('./_data/cifar_double_descent')
    gcs_dir = f'cifar-double-descent_{timestamp()}'
    save_dir_to_gcs(directory, bucket, gcs_dir)
