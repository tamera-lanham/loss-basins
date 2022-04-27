from loss_basins.training_jobs.cifar_double_descent_job import (
    CifarDoubleDescentJob,
    CifarDoubleDescentJobMetadata,
)

if __name__ == "__main__":
    metadata = CifarDoubleDescentJobMetadata()
    job = CifarDoubleDescentJob(metadata)
    job.run()
