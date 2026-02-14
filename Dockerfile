FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends tesseract-ocr libgl1 ghostscript && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -e .
EXPOSE 8000
CMD ["uvicorn", "contract_ai.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
