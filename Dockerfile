FROM python:3.10-slim

WORKDIR /app

# system deps
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY data/raw/churn.csv data/raw/churn.csv

ENV WANDB_MODE=online

CMD ["bash", "-c", "dvc pull && dvc checkout && dvc repro"]



