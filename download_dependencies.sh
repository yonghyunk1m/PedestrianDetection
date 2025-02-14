#!/bin/bash

# Set the pip alias for the current Conda environment
alias my_pip="$(which pip)"

# Install required packages
conda install -y pyyaml
conda install -y pytorch torchvision torchaudio -c pytorch
conda install -y pytorch-lightning
conda install -y transformers
conda install -y -c conda-forge wandb

# Install the fire package using pip
my_pip install fire

echo "✅ All required packages have been successfully installed!"