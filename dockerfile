FROM python:3.11-slim

WORKDIR /app

# Install system dependencies if required for network/compiles
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Set time zone to UK
ENV TZ=Europe/London

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

EXPOSE 5085

CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5085"]