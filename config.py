"""
Configuration file for Person Detection and Identification System
"""

import os

class Config:
    # Model paths and settings
    YOLO_MODEL_PATH = "yolov8n.pt"
    FACE_RECOGNITION_MODEL = "Facenet"  # Options: Facenet, VGG-Face, ArcFace, OpenFace
    FACE_DETECTION_BACKEND = "mtcnn"    # Options: opencv, ssd, dlib, mtcnn, retinaface
    
    # Database settings
    DATABASE_PATH = "face_database/organized"  # Use organized database
    EMBEDDINGS_FILE = os.path.join(DATABASE_PATH, "embeddings.pkl")  # Legacy, not used in organized DB
    METADATA_FILE = os.path.join(DATABASE_PATH, "metadata.json")     # Legacy, not used in organized DB
    
    # Detection thresholds
    PERSON_DETECTION_CONFIDENCE = 0.4
    FACE_DETECTION_CONFIDENCE = 0.9
    FACE_RECOGNITION_THRESHOLD = 0.6  # Lower values = more strict matching
    
    # Processing settings
    INPUT_SIZE = 640  # YOLO input size
    FACE_SIZE = (160, 160)  # Face preprocessing size
    MAX_FACES_PER_PERSON = 5  # Maximum faces to store per person
    
    # Performance settings
    DEVICE = "cpu"  # Options: cpu, cuda, mps
    BATCH_SIZE = 1
    MAX_FPS = 30
    
    # Enrollment settings
    MIN_ENROLLMENT_IMAGES = 3
    MAX_ENROLLMENT_IMAGES = 5
    ENROLLMENT_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp']
    
    # Visualization settings
    BBOX_COLOR_KNOWN = (0, 255, 0)      # Green for known persons
    BBOX_COLOR_UNKNOWN = (0, 0, 255)    # Red for unknown persons
    BBOX_THICKNESS = 2
    TEXT_COLOR = (255, 255, 255)        # White text
    TEXT_FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
    TEXT_SCALE = 0.6
    TEXT_THICKNESS = 2
    
    # Logging settings
    LOG_LEVEL = "INFO"
    LOG_FILE = "person_identification.log"
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        os.makedirs(cls.DATABASE_PATH, exist_ok=True) 