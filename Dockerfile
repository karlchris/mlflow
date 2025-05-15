FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . .

# Defining start-up command
EXPOSE 8080
ENTRYPOINT ["mlflow", "server", "--host", "0.0.0.0", "--port", "8080"]
