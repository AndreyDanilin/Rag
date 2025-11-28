FROM python:3.10-slim

# System deps
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy files
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt
COPY . .

# (Corpus in ./data/corpus expected)
RUN mkdir -p ./data/corpus

ENV PYTHONUNBUFFERED=1

# Entrypoint
CMD ["python", "main.py"]
