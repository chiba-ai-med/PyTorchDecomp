FROM mwalbeck/python-poetry:1-3.12

RUN ls
RUN pwd
RUN poetry install
RUN poetry run python -c "import torchdecomp"
