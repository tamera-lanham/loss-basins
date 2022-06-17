from loss_basins.utils.gcs import *
import shutil


def test_init():
    gcs = GCS("mega-experiment")
    assert isinstance(gcs.bucket, storage.Bucket)
    blobs = list(gcs.bucket.list_blobs(max_results=3))
    assert isinstance(blobs[0], storage.Blob)


def make_test_files(file_structure, output_dir="./_data/test_files"):

    output_dir = Path(output_dir)
    if not output_dir.exists():
        os.makedirs(output_dir)

    def make_files(structure, path):
        for item in structure:
            if isinstance(item, str):
                with open(path / item, "w") as f:
                    f.write(item + " contents")
            else:
                dir_path = path / list(item.keys())[0]
                os.makedirs(dir_path)
                make_files(list(item.values())[0], dir_path)

    make_files(file_structure, output_dir)


def test_upload():
    file_structure = [
        {"dir1": [{"dir3": ["file1", "file2", "file3"]}, "file4", "file5"]},
        {"dir2": ["file6", "file7"]},
    ]

    try:
        output_dir = "./_data/test_files"
        make_test_files(file_structure, output_dir)

        gcs = GCS("mega-experiment")
        gcs.upload(output_dir, "test-folder")

        assert (
            gcs.bucket.get_blob("test-folder/dir1/dir3/file1").download_as_text()
            == "file1 contents"
        )
        assert (
            gcs.bucket.get_blob("test-folder/dir1/file4").download_as_text()
            == "file4 contents"
        )
        assert (
            gcs.bucket.get_blob("test-folder/dir2/file7").download_as_text()
            == "file7 contents"
        )

    finally:
        # Clean up local
        shutil.rmtree(output_dir)

        # Clean up GCS
        blobs = gcs.bucket.list_blobs(prefix="test-folder")
        gcs.bucket.delete_blobs(list(blobs))
