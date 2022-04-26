# sh ./jobs/train-many/run-on-lambda 104.171.200.123

IP=$1
SSH_KEY=~/.ssh/lambda-and-pspace

ssh -i $SSH_KEY ubuntu@$IP "mkdir -p ~/loss-basins"
scp -i $SSH_KEY -r $(pwd)/jobs/ $(pwd)/loss_basins/ $(pwd)/_keys/ $(pwd)/pyproject.toml ubuntu@$IP:~/loss-basins

# scp -i $SSH_KEY -r $(pwd)/jobs/ ubuntu@$IP:~/loss-basins

# This script may or may not work

ssh -i $SSH_KEY ubuntu@$IP "
sudo apt-get update -y; sudo apt-get install -y make build-essential libssl-dev zlib1g-dev \
    libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
    libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
    
curl https://pyenv.run | bash

cat  <<"EOF" >> ~/.bashrc
#pyenv
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
EOF

exec $SHELL

pyenv install 3.9.12
pyenv shell 3.9.12

curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
~/.poetry/bin/poetry env use ~/.pyenv/versions/3.9.12/bin/python
cd ~/loss-basins && ~/.poetry/bin/poetry install
cd ~/loss-basins && ~/.poetry/bin/poetry run python jobs/train-many/train_many.py
"

