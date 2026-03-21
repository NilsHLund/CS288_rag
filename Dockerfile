FROM python:3.10-slim

WORKDIR /app

# Install system dependencies needed by some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first so this layer is cached —
# only re-runs if requirements.txt changes, not your code
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Gradescope: ~4GB RAM, 2 CPU, no GPU — RAG_FAST keeps memory/time within limits (scripts/rag.py).
ENV CS288_RAG_FAST=1

# Default command — overridden at docker run time
CMD ["bash"]