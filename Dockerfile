FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for lightgbm/torch if needed
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- THE FIX: COPY ALL ASSETS ---
# Make sure these filenames match your folder exactly!
COPY mededge_beast_100k.pth .
COPY scaler.pkl .
COPY le_diag.pkl .
COPY le_med.pkl .
COPY app.py .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]