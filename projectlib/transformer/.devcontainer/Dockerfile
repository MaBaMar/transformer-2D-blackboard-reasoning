FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN pip install --no-cache-dir \
    numpy \
    scipy \
    pandas \
    matplotlib \
    seaborn \
    scikit-learn \
    jupyterlab \
    notebook \
    ipykernel \
    tqdm \
    rich \
    pytest \
    black \
    isort \
    mypy \
    torch


CMD ["bash"]
