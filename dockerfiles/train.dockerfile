# Base image
FROM python:3.11-slim AS base

# Install necessary system dependencies
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the necessary files into the container
COPY dockerfiles/train.dockerfile /app/
COPY data/ data/
COPY src/ src/
COPY requirements.txt requirements.txt
COPY requirements_dev.txt requirements_dev.txt
COPY README.md README.md
COPY pyproject.toml pyproject.toml

# Install Python dependencies
RUN pip install -r requirements.txt --no-cache-dir --verbose
RUN pip install . --no-deps --no-cache-dir --verbose

# Set the entrypoint for the container
#ENTRYPOINT ["python", "-m", "cProfile", "-s", "time", "src/animals/train.py"]
ENTRYPOINT ["python", "-u", "src/animals/train.py"]