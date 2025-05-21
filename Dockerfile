FROM python:3.11-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

COPY requirements.txt .
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . .

# Defining start-up command
EXPOSE 8080
ENTRYPOINT ["mlflow", "server", "--host", "0.0.0.0", "--port", "8080"]
