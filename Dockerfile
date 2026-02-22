FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends tesseract-ocr libgl1 ghostscript && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --retries 20 --timeout 1200 \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      torch==2.2.2+cpu torchvision==0.17.2+cpu && \
    pip install --no-cache-dir --retries 20 --timeout 1200 -e .
EXPOSE 8000
CMD ["uvicorn", "contract_ai.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
