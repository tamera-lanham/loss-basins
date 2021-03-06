# Project installation 

First, [install miniconda.](https://docs.conda.io/en/latest/miniconda.html). 

Then clone the repo, and in the repo root directory run the following steps:
```
# Create and activate a conda env for the project: 
conda create -p ./.env/ python=3.10 -y
conda activate ./.env/

# Install torch with conda: 
conda install pytorch torchvision -c pytorch -y

# Install other dependencies into the conda env using poetry: 
poetry install
```

Once you've run this once, you should be able to reenter the proejct environment with `conda activate ./.env` whenever you start a new terminal session. 


If you want to use GCS (Google Cloud Storage), you'll also need to ask Tamera for the GCS key file. Once you have it you should create a directory called `_keys` under the repo root and move the key file into that directory.