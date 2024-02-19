FROM mwalbeck/python-poetry:1-3.12

RUN poetry shell
RUN poetry install
RUN poetry run python -c "import torchdecomp"
