FROM mwalbeck/python-poetry:1-3.12

RUN mkdir my_package &&\
    cd my_package &&\
    poetry init &&\
    poetry install git+https://github.com/chiba-ai-med/PyTorchDecomp.git &&\
    poetry run python -c 'import torchdecomp'
