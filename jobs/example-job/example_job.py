from setuptools_scm import meta
from loss_basins.training_jobs.example_training_job import ExampleJobMetadata
from loss_basins.training_jobs.example_training_job import (
    ExampleTrainingJob,
    ExampleJobMetadata,
)

if __name__ == "__main__":
    metadata = ExampleJobMetadata(n_init_repeats=5, gcs_bucket="mega-experiment")
    print(metadata)
    job = ExampleTrainingJob(metadata)
    job.run()
