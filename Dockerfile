FROM mwalbeck/python-poetry:1-3.12

ADD poetry.lock /PyTorchDecomp/
ADD pyproject.toml /PyTorchDecomp/
ADD LICENSE /PyTorchDecomp/
ADD README.md /PyTorchDecomp/
ADD torchdecomp /PyTorchDecomp/
ADD tests /PyTorchDecomp/

WORKDIR /PyTorchDecomp

RUN ls
RUN pwd
RUN poetry install -vvv && poetry run python -c 'import torchdecomp'
