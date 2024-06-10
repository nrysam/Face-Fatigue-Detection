# Face Fatigue Detection
## Overview
Face Fatigue Detection is an application that uses a combination of face recognition and fatigue detection models to identify users and determine their fatigue status based on video input from a webcam. The system utilizes YOLOv5 for face detection and a ResNet-based model for fatigue classification.


## Features
- User registration with face image capture
- Face recognition for registered users
- Fatigue detection using a pre-trained deep learning model
- Real-time video processing with OpenCV


## Requirements
- Python 3.9
- Docker
- MySQL database
- Host system with webcam support


## Installation
### Clone the Repository

```bash
git clone https://github.com/nrysam/Face-Fatigue-Detection
cd Face-Fatigue-Detection
```


## Set Up the Environment
### Install Dependencies

```bat
pip install -r requirements.txt
```


### Database Setup
Create a MySQL database and update the .env file with your database credentials:
```sql
DB_USER=your_db_user
DB_PASS=your_db_password
DB_HOST=your_db_host
DB_NAME=your_db_name
```
Remember to replace the user, password, host, and name with the actual database credentials. 


### Docker Setup
**Build the Docker Image**

```cmd
docker build -t face_fatigue_detection .
```


**Run the Docker Container**
```cmd
docker run --rm -it \
    --privileged \
    --device /dev/video0:/dev/video0 \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v "/path/to/your/local/registered_faces:/usr/src/app/registered_faces" \
    face_fatigue_detection
```

Replace **'/path/to/your/local/registered_faces'** with the actual path to the directory on your host machine where you want to store the registered face images.


## Usage
### Register a New User
When you run the application, you will be prompted to register if you haven't done so already.

### Follow the prompts to enter your name and email.
`Press 's' to capture your face image for registration.`

### Face Recognition and Fatigue Detection
If you are already registered, the application will recognize you and proceed with fatigue detection.

- The system will start capturing video from the webcam.
- Detected faces will be displayed with bounding boxes.
- Fatigue status will be displayed above each detected face.

`Press 'q' to exit the application.`


## Files and Directories
- face_fatigue_detection.py: Main script for the application.
- Dockerfile: Dockerfile to build the application image.
- requirements.txt: Python dependencies.
- registered_faces/: Directory to store registered face images.


## Troubleshooting
### Webcam Access Issues
If you encounter issues with webcam access in the Docker container, ensure that:

- The correct camera index is used.
- The host system's webcam is accessible.
- Docker has the necessary permissions to access /dev/video0.


## X Server Access
For GUI applications, ensure that your Docker container can access the X server on the host system:
```cmd
xhost +local:root
```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.


## License
This project is licensed under the Apache 2.0 License.
