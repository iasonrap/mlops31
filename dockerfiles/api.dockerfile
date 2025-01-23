# Change from latest to a specific version if your requirements.txt
FROM python:3.11-slim AS base

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY src/ src/
COPY requirements_backend.txt requirements_backend.txt
COPY README.md README.md
COPY pyproject.toml pyproject.toml

RUN pip install -r requirements_backend.txt --no-cache-dir --verbose
RUN pip install . --no-deps --no-cache-dir --verbose


EXPOSE $PORT
CMD exec uvicorn src.animals.api:app --host 0.0.0.0 --port $PORT --workers 1 
