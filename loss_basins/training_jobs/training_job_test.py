from loss_basins.training_jobs.training_job import *
from loss_basins.training_jobs.example_training_job import *
import shutil


def test_training_job():
    job_metadata = ExampleJobMetadata(
        n_inputs=8, n_batches=100, epochs=5, n_init_repeats=5
    )
    job = ExampleTrainingJob(job_metadata)
    job.run()

    assert job.output_path.exists()
    assert (job.output_path / "inits").exists()

    assert len(list((job.output_path / "inits").iterdir())) == 5

    for init in (job.output_path / "inits").iterdir():
        parameter_checkpoints = (init / "parameter_checkpoints").iterdir()
        assert len(list(parameter_checkpoints)) == job.metadata.epochs + 1

    # Cleanup
    shutil.rmtree(job.output_path)


if __name__ == "__main__":
    test_training_job()
