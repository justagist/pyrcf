FROM mcr.microsoft.com/devcontainers/python:0-3.10

RUN apt-get update && apt install git-lfs

RUN python -m pip install --upgrade pip \
    && python -m pip install 'flit>=3.8.0'

ENV FLIT_ROOT_INSTALL=1

COPY pyproject.toml .
RUN touch README.md \
    && mkdir -p src/python_template \
    && python -m flit install --only-deps --deps develop \
    && rm -r pyproject.toml README.md src
