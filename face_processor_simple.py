"""
Simplified Face Processing Module
Uses OpenCV for face detection and basic feature extraction
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging
from config import Config

logger = logging.getLogger(__name__)

class SimpleFaceProcessor:
    """
    Simplified face detection and recognition using OpenCV
    """
    
    def __init__(self):
        """Initialize face processor with OpenCV cascade classifier"""
        try:
            # Load OpenCV's Haar cascade for face detection
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Initialize ORB feature detector for face recognition
            self.orb = cv2.ORB_create(nfeatures=500)
            
            logger.info("Simple face processor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize face processor: {e}")
            raise
    
    def detect_faces(self, image: np.ndarray, min_confidence: float = None) -> List[Dict[str, Any]]:
        """
        Detect faces in the given image using OpenCV
        
        Args:
            image: Input image
            min_confidence: Not used in this simple implementation
            
        Returns:
            List of face detections with bounding boxes
        """
        try:
            # Convert to grayscale for face detection
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            detections = []
            for (x, y, w, h) in faces:
                detections.append({
                    'bbox': (x, y, x + w, y + h),
                    'confidence': 1.0,  # OpenCV doesn't provide confidence scores
                    'face_img': None
                })
            
            return detections
            
        except Exception as e:
            logger.error(f"Error in face detection: {e}")
            return []
    
    def extract_face_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face embedding using ORB features
        
        Args:
            face_image: Cropped face image
            
        Returns:
            Face embedding vector or None if extraction fails
        """
        try:
            # Convert to grayscale
            if len(face_image.shape) == 3:
                gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            else:
                gray = face_image
            
            # Resize to standard size
            gray = cv2.resize(gray, (128, 128))
            
            # Extract ORB features
            keypoints, descriptors = self.orb.detectAndCompute(gray, None)
            
            if descriptors is not None and len(descriptors) > 0:
                # Create a fixed-size embedding from descriptors
                # Use histogram of descriptors as a simple embedding
                embedding = np.zeros(512)  # Fixed size embedding
                
                # Fill embedding with descriptor statistics
                desc_flat = descriptors.flatten()
                if len(desc_flat) > 0:
                    # Use first 256 values directly
                    end_idx = min(256, len(desc_flat))
                    embedding[:end_idx] = desc_flat[:end_idx]
                    
                    # Add statistical features
                    embedding[256] = np.mean(desc_flat)
                    embedding[257] = np.std(desc_flat)
                    embedding[258] = np.min(desc_flat)
                    embedding[259] = np.max(desc_flat)
                    embedding[260] = len(keypoints)
                    
                    # Fill remaining with histogram of descriptor values
                    hist, _ = np.histogram(desc_flat, bins=251, range=(0, 255))
                    embedding[261:512] = hist
                
                return embedding
            else:
                # Fallback: use image statistics as embedding
                embedding = np.zeros(512)
                
                # Basic image statistics
                embedding[0] = np.mean(gray)
                embedding[1] = np.std(gray)
                embedding[2] = np.min(gray)
                embedding[3] = np.max(gray)
                
                # Histogram features
                hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
                hist = hist.flatten()
                embedding[4:260] = hist
                
                # LBP-like features (simplified)
                lbp_hist = self._simple_lbp_histogram(gray)
                embedding[260:512] = lbp_hist[:252]
                
                return embedding
                
        except Exception as e:
            logger.error(f"Error extracting face embedding: {e}")
            return None
    
    def _simple_lbp_histogram(self, image: np.ndarray) -> np.ndarray:
        """
        Compute a simplified Local Binary Pattern histogram
        
        Args:
            image: Grayscale image
            
        Returns:
            LBP histogram
        """
        try:
            h, w = image.shape
            lbp_values = []
            
            # Simple LBP computation (3x3 neighborhood)
            for i in range(1, h-1):
                for j in range(1, w-1):
                    center = image[i, j]
                    binary_string = ""
                    
                    # Compare with 8 neighbors
                    neighbors = [
                        image[i-1, j-1], image[i-1, j], image[i-1, j+1],
                        image[i, j+1], image[i+1, j+1], image[i+1, j],
                        image[i+1, j-1], image[i, j-1]
                    ]
                    
                    for neighbor in neighbors:
                        binary_string += '1' if neighbor >= center else '0'
                    
                    lbp_values.append(int(binary_string, 2))
            
            # Create histogram
            hist, _ = np.histogram(lbp_values, bins=256, range=(0, 255))
            return hist.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error computing LBP histogram: {e}")
            return np.zeros(256, dtype=np.float32)
    
    def extract_faces_from_person(self, person_image: np.ndarray) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
        """
        Extract all faces from a person's bounding box region
        
        Args:
            person_image: Cropped person region
            
        Returns:
            List of (face_image, detection_info) tuples
        """
        try:
            detections = self.detect_faces(person_image)
            faces = []
            
            for detection in detections:
                bbox = detection['bbox']
                x1, y1, x2, y2 = bbox
                
                # Ensure coordinates are within image bounds
                h, w = person_image.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 > x1 and y2 > y1:
                    face_img = person_image[y1:y2, x1:x2]
                    if face_img.size > 0:
                        faces.append((face_img, detection))
            
            return faces
            
        except Exception as e:
            logger.error(f"Error extracting faces from person: {e}")
            return []
    
    def compare_embeddings(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate similarity between two face embeddings
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score (0-1, higher is more similar)
        """
        try:
            # Calculate cosine similarity
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            cosine_sim = dot_product / (norm1 * norm2)
            # Convert to 0-1 range (cosine similarity is -1 to 1)
            similarity = (cosine_sim + 1) / 2
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error comparing embeddings: {e}")
            return 0.0
    
    def preprocess_face(self, face_image: np.ndarray, target_size: Tuple[int, int] = None) -> np.ndarray:
        """
        Preprocess face image for recognition
        
        Args:
            face_image: Input face image
            target_size: Target size for resizing
            
        Returns:
            Preprocessed face image
        """
        try:
            target_size = target_size or Config.FACE_SIZE
            
            # Resize image
            face_resized = cv2.resize(face_image, target_size)
            
            # Normalize pixel values
            face_normalized = face_resized.astype(np.float32) / 255.0
            
            return face_normalized
            
        except Exception as e:
            logger.error(f"Error preprocessing face: {e}")
            return face_image 