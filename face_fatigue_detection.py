import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
from PIL import Image
from sqlalchemy import create_engine, MetaData, Column, Integer, String, Enum, TIMESTAMP, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
import enum
import os
import pickle
import datetime

# Database setup
Base = declarative_base()

class User(Base):
    __tablename__ = 'Users'
    user_id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(50), nullable=False)  
    email = Column(String(100), unique=True, nullable=False)  
    created_at = Column(TIMESTAMP, default=datetime.datetime.utcnow)

class FaceRecognition(Base):
    __tablename__ = 'FaceRecognition'
    recognition_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('Users.user_id'))
    image_path = Column(String(255), nullable=False)  
    detected_at = Column(TIMESTAMP, default=datetime.datetime.utcnow)
    user = relationship("User", back_populates="face_recognitions")
    fatigue_analyses = relationship("FatigueAnalysis", back_populates="recognition")

class FatigueStatus(enum.Enum):
    active = "active"
    fatigue = "fatigue"

class FatigueAnalysis(Base):
    __tablename__ = 'FatigueAnalysis'
    analysis_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('Users.user_id'))
    recognition_id = Column(Integer, ForeignKey('FaceRecognition.recognition_id'))
    status = Column(Enum(FatigueStatus), nullable=False)
    analyzed_at = Column(TIMESTAMP, default=datetime.datetime.utcnow)
    user = relationship("User", back_populates="fatigue_analyses")
    recognition = relationship("FaceRecognition", back_populates="fatigue_analyses")

class Log(Base):
    __tablename__ = 'Logs'
    log_id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('Users.user_id'))
    message = Column(String(500), nullable=False)  
    logged_at = Column(TIMESTAMP, default=datetime.datetime.utcnow)
    user = relationship("User", back_populates="logs")

User.face_recognitions = relationship("FaceRecognition", order_by=FaceRecognition.recognition_id, back_populates="user")
User.fatigue_analyses = relationship("FatigueAnalysis", order_by=FatigueAnalysis.analysis_id, back_populates="user")
User.logs = relationship("Log", order_by=Log.log_id, back_populates="user")

# Database connection
# Define the database connection URL
engine = create_engine(f'mysql+pymysql://{os.getenv("DB_USER")}:{os.getenv("DB_PASS")}@{os.getenv("DB_HOST")}/{os.getenv("DB_NAME")}')

# Create all tables defined by Base's subclasses
Base.metadata.create_all(engine)

# Creating a session
Session = sessionmaker(bind=engine)
session = Session()

# Load the pre-trained YOLOv5 model for face detection
model_face = torch.hub.load('ultralytics/yolov5', 'yolov5s') 

# Load the trained fatigue detection model
class_names = ['active', 'fatigue']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
fatigue_model = models.resnet18()
num_ftrs = fatigue_model.fc.in_features
fatigue_model.fc = nn.Linear(num_ftrs, 2)
fatigue_model.load_state_dict(torch.load('fatigue_detection_model.pth'))
fatigue_model = fatigue_model.to(device)
fatigue_model.eval()

# Define image pre-processing transformations
preprocess = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Function to detect face and predict fatigue
def detect_and_predict(frame):
    results = model_face(frame)

    # Debug: Print YOLOv5 results
    print(f"YOLOv5 Results: {results.xyxy[0]}")
    
    predictions = []
    
    for *box, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, box)
        face = frame[y1:y2, x1:x2]

        # Debug: Print the detected face shape
        print(f"Detected Face Shape: {face.shape}")

        # Pre-process the face image
        face_pil = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_pil)
        face_tensor = preprocess(face_pil).unsqueeze(0).to(device)

        # Predict fatigue state
        with torch.no_grad():
            outputs = fatigue_model(face_tensor)
            _, preds = torch.max(outputs, 1)
            pred_class = class_names[preds[0]]

        predictions.append((x1, y1, x2, y2, pred_class))
    
    return predictions

# Function to register a new user
def register_user():
    username = input("Enter your name: ")
    email = input("Enter your email: ")

    # Check if the email already exists in the database
    if session.query(User).filter_by(email=email).first():
        print("This email is already registered. Please use a different email.")
        return False
        
    new_user = User(username=username, email=email)
    session.add(new_user)
    try:
        session.commit()
    except IntegrityError as e:
        print("This email is already registered. Please use a different email.")
        session.rollback()  # Rollback the session to avoid any corrupt state
        return False

    user_id = new_user.user_id

    # Capture face for recognition
    cap = cv2.VideoCapture(0) # Use either 0 or 1 for using integrated webcam or usb camera
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('Face Registration', frame)
        if cv2.waitKey(1) & 0xFF == ord('s'):  # Press 's' to save the image
            image_path = f"registered_faces/{username}.jpg"
            cv2.imwrite(image_path, frame)
            new_recognition = FaceRecognition(user_id=user_id, image_path=image_path)
            session.add(new_recognition)
            session.commit()
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Registration complete.")

# Function to check if user is registered
def is_registered():
    email = input("Enter your email to check registration: ")
    user = session.query(User).filter_by(email=email).first()
    return user

# Main program
if __name__ == "__main__":
    if not os.path.exists('registered_faces'):
        os.makedirs('registered_faces')

    while True:
        response = input("Have you registered yet? (yes/no): ").strip().lower()
        if response == 'no':
            register_user()
        elif response == 'yes':
            user = is_registered()
            if user:
                break
            else:
                print("User not found. Please register first.")
        else:
            print("Please answer 'yes' or 'no'.")

    # Proceed with fatigue detection
    print(f"Hello, {user.username}!")

    # Open a video stream
    cap = cv2.VideoCapture(0)

    cv2.namedWindow('Fatigue Detection', cv2.WINDOW_NORMAL)  # Create a resizable window
    cv2.resizeWindow('Fatigue Detection', 640, 480)  # Set the window size

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        predictions = detect_and_predict(frame)

        # Draw bounding boxes and labels on the frame
        for (x1, y1, x2, y2, pred_class) in predictions:
            color = (0, 255, 0) if pred_class == 'active' else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, pred_class, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow('Fatigue Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to close the windows
            break

    cap.release()
    cv2.destroyAllWindows()