# Base image -- must match the Python version used in development.
# The project uses modern syntax (PEP 604 unions, etc.) that requires >= 3.10.
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# ffmpeg + ffprobe are required for all audio transcoding and probing.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY scripts ./scripts
COPY .env.example ./.env.example

EXPOSE 8000

# Bind to 0.0.0.0 so the container port is reachable from the host.
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
