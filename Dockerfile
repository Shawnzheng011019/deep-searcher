FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY uv.lock ./

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir uv

RUN uv pip install --system -r pyproject.toml

COPY . .

RUN pip install --no-cache-dir -e .

RUN uv pip install --system tavily-python>=0.5.0

RUN mkdir -p /app/data /app/logs

EXPOSE 8000

ENV PYTHONPATH=/app
ENV MILVUS_DATA_PATH=/app/data

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/docs || exit 1

CMD ["python", "main.py", "--enable-cors", "true", "--enable-web-search"] 