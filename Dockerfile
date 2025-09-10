FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libfreetype6-dev \
        libpng-dev \
        libjpeg-dev \
    && pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY ./app/code ./code
COPY ./models ./models

RUN mkdir -p ./data
COPY ./data ./data

EXPOSE 8050

CMD ["python", "code/app.py"]
