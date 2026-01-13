#!/bin/bash

python3.12 -m venv .venv
source .venv/bin/activate

pip install torch torchvision torchaudio
pip install wandb
pip install tqdm
pip install transformers
pip install pandas
pip install matplotlib
pip install pytest
