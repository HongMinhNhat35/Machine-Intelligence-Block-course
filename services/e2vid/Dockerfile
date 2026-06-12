FROM python:3.12.3-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 300 \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      -r requirements.txt && \
    pip install --no-cache-dir opencv-python-headless

COPY app.py .
COPY weights/ /app/weights/

EXPOSE 8001
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
