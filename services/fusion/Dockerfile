FROM python:3.12.3-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl libxcb1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY fusion_layer.py .
COPY run_pipeline.py .
COPY weights/ /app/weights/

EXPOSE 8003
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8003"]
