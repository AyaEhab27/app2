FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y espeak-ng

COPY app/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app/ .

COPY app/models/ ./models/

COPY app/voice-ec9bd-firebase-adminsdk-fbsvc-0215fa1324.json ./voice-ec9bd-firebase-adminsdk-fbsvc-0215fa1324.json

EXPOSE 80

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
