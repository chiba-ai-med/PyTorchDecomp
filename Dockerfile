FROM mwalbeck/python-poetry:1-3.12

RUN apt-get update &&\
    apt-get install git -y &&\
    git clone https://github.com/chiba-ai-med/PyTorchDecomp.git &&\
    cd PyTorchDecomp &&\
    pip install dist/torchdecomp-0.1.0-py3-none-any.whl &&\
    pip install pytest-cov &&\
    pytest --cov=torchdecomp
