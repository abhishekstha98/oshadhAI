
# 1. Setup System as Root
FROM python:3.10-slim

# Install system dependencies for RDKit
RUN apt-get update && apt-get install -y \
    libxrender1 \
    libxext6 \
    libsm6 \
    && rm -rf /var/lib/apt/lists/*

# 2. Setup Non-Root User (Hugging Face Security Requirement)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# Set working directory to user's home
WORKDIR /home/user/app

# 3. Install Dependencies
COPY --chown=user requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# 4. Copy Application Code
COPY --chown=user src/ src/
COPY --chown=user data/ data/
COPY --chown=user checkpoints/ checkpoints/

# 5. Run Configuration
EXPOSE 7860
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "7860"]
