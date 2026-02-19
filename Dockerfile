# ============================================================================
# Dockerfile
# Decision-Centric ML Customer Retention — Production Container
# ============================================================================
FROM python:3.11-slim

# Set labels
LABEL maintainer="MachinelearningNCKH"
LABEL description="Decision-Centric Survival Analysis Pipeline + Streamlit Dashboard"

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Create working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgomp1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer caching)
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy project source (data/ excluded via .dockerignore)
COPY src/         ./src/
COPY tests/       ./tests/
COPY config/      ./config/
COPY app.py       ./app.py
COPY main.py      ./main.py
COPY README.md    ./README.md

# Create output directories
RUN mkdir -p outputs data/raw data/processed

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command: launch dashboard
# To run pipeline instead: docker run <image> python main.py --no-shap
CMD ["streamlit", "run", "app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true"]
