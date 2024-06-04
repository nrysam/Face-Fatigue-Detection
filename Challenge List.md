## 1. Database Design (Status: Done)
### Description:
The project utilizes an SQLAlchemy-based ORM for database design. Three tables are defined: Users, FaceRecognition, and FatigueAnalysis. These tables store user information, registered face images, and fatigue analysis results, respectively.

### Problem encountered:
None.

## 2. Dataset Collection (Status: Done)
### Description:
The dataset for training, validation, and testing was collected from the University of Texas at Arlington Real-Life Drowsiness Dataset (UTA-RLDD) available on Kaggle. The dataset includes labeled images categorized into training, validation, and test sets. Data transformations applied include resizing images to 128x128 pixels, normalization, and random horizontal flipping for the training set.

### Problem encountered:
None.

## 3. Face Recognition (Status: Done)
### Description:
The project uses the YOLOv5 model for face detection. The captured faces are registered and compared with stored images for user recognition.
### Problem encountered:
- Ensuring that the Docker container correctly uses the intended camera device. This requires correctly mapping the device using the **'--device'** flag 

- Ensuring permissions and camera index are properly set in the python code.

  `cap = cv2.VideoCapture(0)  # Change to 1 if using an external USB camera`

## 4. Fatigue Analysis (Status: Done)
### Description:
A ResNet18 model, pre-trained on ImageNet, was fine-tuned for fatigue detection. The model was trained, validated, and tested on a custom dataset. The model predicts whether a detected face shows signs of fatigue.

### Problem encountered:
None.

## 5. Integration (Status: Done)
### Description:
The system integrates face recognition and fatigue analysis modules. Detected faces are analyzed for fatigue, and results are stored in the database.

### Problem encountered:
None.

## 6. Deployment (Status: Done)
### Description:
The application is containerized using Docker for easy deployment. The Dockerfile sets up the necessary environment and dependencies.

### Problem encountered:
Handling camera access in Docker container. Solved by sharing the device.
