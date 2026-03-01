FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# TODO: expose correct port / transport once server.py is implemented
CMD ["python", "src/server.py"]
