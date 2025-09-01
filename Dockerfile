FROM python:3.11-slim

# Install system dependencies for building packages
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip and setuptools early
RUN pip install --upgrade pip setuptools wheel

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

EXPOSE 5000
CMD ["python", "app.py"]
