# Installation on Lambda

```
# Create and activate conda env: 
conda create -p ./.conda-env/ python=3.9 -y
conda activate ./.conda-env/

# Install torch with conda: 
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -y

# Install dependencies into conda env using poetry: 
poetry install
```