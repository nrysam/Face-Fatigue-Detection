# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Install system dependencies for GUI and OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1-mesa-glx \
        libglib2.0-0 \
        x11-apps \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code
COPY . .

# Set environment variable for the display, pointing to the host's X server
ENV DISPLAY=host.docker.internal:0.0

# Define environment variables for database connectivity
ENV DB_USER=HenrySM
ENV DB_PASS=henry70
ENV DB_HOST=host.docker.internal
ENV DB_NAME=FatigueDetection

# Run face_fatigue_detection.py when the container launches
CMD ["python", "face_fatigue_detection.py"]
