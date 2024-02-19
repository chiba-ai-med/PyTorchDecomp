FROM continuumio/miniconda3

RUN python -m venv env
RUN pip install git+https://github.com/chiba-ai-med/PyTorchDecomp.git
RUN pip install pytest
RUN python -c "import pytest; import importlib.resources; import os; pytest.main([os.path.join(importlib.resources.files('torchdecomp'), 'tests')])"
