# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# تثبيت الحزم الأساسية
RUN apt-get update && \
    apt-get install -y espeak-ng

# Copy the requirements file into the container
COPY app/requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY app/ .

# Copy the models directory
COPY app/models/ ./models/

# Copy the Firebase service account key
COPY app/voice-ec9bd-firebase-adminsdk-fbsvc-0215fa1324.json ./voice-ec9bd-firebase-adminsdk-fbsvc-0215fa1324.json

# Make port 80 available to the world outside this container
EXPOSE 80

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
