FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY data/processed/ data/processed/

CMD ["python", "src/run_monte_carlo_benchmarks.py"]
