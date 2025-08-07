"""
Person Detection Module using YOLOv8
Enhanced version of the original detector with additional functionality
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Optional
from config import Config
import logging

logger = logging.getLogger(__name__)

class PersonDetector:
    """
    Person detection using YOLOv8 model
    """
    
    def __init__(self, model_path: str = None, confidence: float = None, device: str = None):
        """
        Initialize the person detector
        
        Args:
            model_path: Path to YOLO model file
            confidence: Detection confidence threshold
            device: Device to run inference on (cpu, cuda, mps)
        """
        self.model_path = model_path or Config.YOLO_MODEL_PATH
        self.confidence = confidence or Config.PERSON_DETECTION_CONFIDENCE
        self.device = device or Config.DEVICE
        
        try:
            self.model = YOLO(self.model_path)
            logger.info(f"YOLO model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    def detect_persons(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect persons in the given frame
        
        Args:
            frame: Input image/frame
            
        Returns:
            List of detections: [(x1, y1, x2, y2, confidence), ...]
        """
        try:
            results = self.model.predict(
                source=frame,
                conf=self.confidence,
                save=False,
                imgsz=Config.INPUT_SIZE,
                device=self.device,
                classes=[0],  # Only detect persons (class 0 in COCO)
                verbose=False
            )
            
            detections = []
            if results and len(results) > 0 and results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    detections.append((x1, y1, x2, y2, conf))
            
            return detections
            
        except Exception as e:
            logger.error(f"Error in person detection: {e}")
            return []
    
    def crop_person_region(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                          padding: float = 0.1) -> Optional[np.ndarray]:
        """
        Crop person region from frame with optional padding
        
        Args:
            frame: Input frame
            bbox: Bounding box (x1, y1, x2, y2)
            padding: Padding factor (0.1 = 10% padding)
            
        Returns:
            Cropped image or None if invalid
        """
        try:
            x1, y1, x2, y2 = bbox
            h, w = frame.shape[:2]
            
            # Add padding
            width = x2 - x1
            height = y2 - y1
            pad_w = int(width * padding)
            pad_h = int(height * padding)
            
            # Ensure coordinates are within frame bounds
            x1 = max(0, x1 - pad_w)
            y1 = max(0, y1 - pad_h)
            x2 = min(w, x2 + pad_w)
            y2 = min(h, y2 + pad_h)
            
            if x2 <= x1 or y2 <= y1:
                return None
                
            return frame[y1:y2, x1:x2]
            
        except Exception as e:
            logger.error(f"Error cropping person region: {e}")
            return None
    
    def visualize_detections(self, frame: np.ndarray, 
                           detections: List[Tuple[int, int, int, int, float]],
                           labels: List[str] = None) -> np.ndarray:
        """
        Draw detection boxes and labels on frame
        
        Args:
            frame: Input frame
            detections: List of detections
            labels: Optional labels for each detection
            
        Returns:
            Frame with visualizations
        """
        frame_copy = frame.copy()
        
        for i, (x1, y1, x2, y2, conf) in enumerate(detections):
            # Choose color based on whether we have a label
            if labels and i < len(labels) and labels[i] != "Unknown":
                color = Config.BBOX_COLOR_KNOWN
                label = labels[i]
            else:
                color = Config.BBOX_COLOR_UNKNOWN
                label = "Unknown"
            
            # Draw bounding box
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, Config.BBOX_THICKNESS)
            
            # Draw label with confidence
            label_text = f"{label} ({conf:.2f})"
            label_size = cv2.getTextSize(label_text, Config.TEXT_FONT, 
                                       Config.TEXT_SCALE, Config.TEXT_THICKNESS)[0]
            
            # Draw label background
            cv2.rectangle(frame_copy, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(frame_copy, label_text, (x1, y1 - 5),
                       Config.TEXT_FONT, Config.TEXT_SCALE, 
                       Config.TEXT_COLOR, Config.TEXT_THICKNESS)
        
        return frame_copy 