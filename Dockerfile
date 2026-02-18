# ==========================================
# STAGE 1: The Builder (Compiles & Installs)
# ==========================================

FROM python:3.12-slim AS builder

# Prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 1. Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

# 2. Create a virtual environment
RUN python -m venv /opt/venv
# Enable the venv for the next commands
ENV PATH="/opt/venv/bin:$PATH"

# 3. Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# ==========================================
# STAGE 2: The Runner (Production Image)
# ==========================================
# We start fresh with a clean Python 3.12 image
FROM python:3.12-slim

WORKDIR /app

# 1. Copy ONLY the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# 2. Enable the venv in this stage too
ENV PATH="/opt/venv/bin:$PATH"

# 3. Copy your application code
COPY . .

# 4. Expose the port
EXPOSE 8000

# 5. The command to run the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]