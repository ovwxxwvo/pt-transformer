# Base image: Python-slim
FROM python:3.13-slim

# Set working directory (path inside the container)
WORKDIR /app

# Install system dependencies (resolve compilation issues for some Python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency list to the container
COPY requirements.txt .

# Install Python dependencies (use Tsinghua mirror for acceleration, avoid timeout)
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Copy the entire project code to the container
COPY . .
RUN chmod +x /app/server.sh

# Create data/log directories (avoid permission errors)
RUN mkdir -p /app/data /app/logs /app/database && chmod 777 /app/data /app/logs /app/database

# Expose ports (8000 for API, 7860 for WebUI)
EXPOSE 8000 7860

# Default command: start API service
CMD ["bash", "server.sh"]


