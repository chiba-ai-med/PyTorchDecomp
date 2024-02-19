FROM continuumio/miniconda3

RUN python -m venv env

RUN pip install git+https://github.com/chiba-ai-med/PyTorchDecomp.git

RUN python -c 'import torchdecomp'
