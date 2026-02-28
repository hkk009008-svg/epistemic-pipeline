FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

ENV PORT=8000

CMD ["sh", "-c", "echo \"Starting uvicorn on port $PORT\" && exec uvicorn app:app --host 0.0.0.0 --port $PORT"]
