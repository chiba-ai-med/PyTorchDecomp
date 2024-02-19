FROM mwalbeck/python-poetry:1-3.12

ADD poetry.lock /
ADD pyproject.toml /
ADD LICENSE /
ADD README.md /
ADD torchdecomp /
ADD tests /

WORKDIR /

RUN poetry install -vvv
RUN poetry run python -c "import torchdecomp"
