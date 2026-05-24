FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH="/app"
ENV PIP_NO_BUILD_ISOLATION=1

WORKDIR /app

# System deps: build tools for mamba-ssm + geospatial stack
RUN apt-get update && apt-get install -y \
    git build-essential ninja-build \
    libgeos-dev libproj-dev proj-data proj-bin \
    libgdal-dev gdal-bin \
    libeccodes-dev \
    && rm -rf /var/lib/apt/lists/*

# Python deps — two-step: base packages first, then mamba-ssm (needs CUDA headers)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir --no-build-isolation \
    causal-conv1d==1.4.0 \
    mamba-ssm==2.2.2

# Project code (no data, no experiments — mounted at runtime)
COPY config/ ./config/
COPY src/     ./src/
COPY scripts/ ./scripts/

RUN mkdir -p /app/data /app/experiments
