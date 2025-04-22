FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

# Cập nhật pip trước khi cài đặt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
    RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libgl1-mesa-glx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY . .

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
