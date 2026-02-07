FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for RDKit
# RDKit often needs libxrender1, libxext6 etc.
RUN apt-get update && apt-get install -y \
    libxrender1 \
    libxext6 \
    libsm6 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY data/ data/
COPY checkpoints/ checkpoints/

# Expose port (Hugging Face Spaces defaults to 7860)
EXPOSE 7860

# Run API
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "7860"]
