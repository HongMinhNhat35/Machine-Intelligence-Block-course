FROM python:3.12.3-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      fastapi==0.111.0 uvicorn==0.30.1 ultralytics==8.4.54 \
      torch==2.3.0+cpu torchvision==0.18.0+cpu
RUN pip uninstall -y opencv-python opencv-python-headless 2>/dev/null; \
    pip install --no-cache-dir opencv-python-headless

COPY app.py .
COPY weights/ /app/weights/

EXPOSE 8001
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]
