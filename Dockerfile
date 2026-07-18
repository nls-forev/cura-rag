FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    PORT=7860 \
    QDRANT_PATH=/data/qdrant \
    SEED_ON_STARTUP=true \
    SEED_INCLUDE_OPENFDA=false

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

# CPU-only torch keeps the image gigabytes smaller; the API container has no GPU.
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install ".[frontend]"

# Bake the embedding and reranker models into the image so the container is
# queryable without a model download at first request.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY data ./data
COPY scripts ./scripts
COPY eval ./eval

# Writable by any UID: Hugging Face Spaces runs the container as a non-root user,
# and the embedded Qdrant index is written under /data at runtime.
RUN mkdir -p /data && chmod 777 /data /models

EXPOSE 7860

# No API key is baked in. The LLM key (e.g. DEEPSEEK_API_KEY) is supplied at
# runtime as an environment variable / platform secret.
CMD ["sh", "-c", "uvicorn curarag.api.main:app --host 0.0.0.0 --port ${PORT}"]
