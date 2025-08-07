#!/usr/bin/env python3
"""
Simple Person Identification System with Organized Database
Uses the new organized database structure for better performance and scalability
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging
import time
import argparse
from config import Config
from person_detector import PersonDetector
from face_processor_simple import SimpleFaceProcessor
from face_database_organized import OrganizedFaceDatabase
from attention_tracker import AttentionTracker
# Setup logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimplePersonIdentificationSystemOrganized:
    """
    Simple person identification system using organized database
    """
    
    def __init__(self):
        """Initialize the identification system"""
        try:
            # Initialize components
            self.person_detector = PersonDetector()
            self.face_processor = SimpleFaceProcessor()
            self.database = OrganizedFaceDatabase()
            self.attention_tracker = AttentionTracker()
            # Performance tracking
            self.fps_counter = 0
            self.fps_start_time = time.time()
            self.current_fps = 0.0
            
            logger.info("Organized Person Identification System initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize system: {e}")
            raise
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Process a single frame for person detection and identification
        
        Args:
            frame: Input video frame
            
        Returns:
            (processed_frame, detections_info)
        """
        try:
            # Step 1: Detect persons in the frame
            person_detections = self.person_detector.detect_persons(frame)
            
            if not person_detections:
                return frame, []
            
            # Step 2: Process each detected person
            identification_results = []
            
            for i, (x1, y1, x2, y2, conf) in enumerate(person_detections):
                # Crop person region
                person_crop = self.person_detector.crop_person_region(frame, (x1, y1, x2, y2))
                
                if person_crop is None:
                    continue
                
                # Step 3: Extract faces from person region
                faces = self.face_processor.extract_faces_from_person(person_crop)
                
                # Analysing the attention of the person 
                attention_data = self.attention_tracker.analyze_frame(person_crop)
                attention_score = attention_data['score'] if attention_data else None
                person_info = {
                    'person_id': i,
                    'bbox': (x1, y1, x2, y2),
                    'confidence': conf,
                    'identity': 'Unknown',
                    'person_db_id': None,
                    'face_similarity': 0.0,
                    'num_faces': len(faces),
                    'attention_score': attention_score
                }
                
                if faces:
                    # Use the largest face
                    best_face = max(faces, key=lambda x: x[0].shape[0] * x[0].shape[1])
                    face_img, face_detection = best_face
                    
                    # Step 4: Extract face embedding
                    face_embedding = self.face_processor.extract_face_embedding(face_img)
                    
                    if face_embedding is not None:
                        # Step 5: Identify person from organized database
                        identity, person_db_id, similarity = self.database.identify_person(face_embedding)
                        
                        if identity:
                            person_info['identity'] = identity
                            person_info['person_db_id'] = person_db_id
                            person_info['face_similarity'] = similarity
                
                identification_results.append(person_info)
            
            # Step 6: Visualize results
            processed_frame = self._visualize_results(frame, identification_results)
            
            # Update FPS counter
            self._update_fps()
            
            return processed_frame, identification_results
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return frame, []
    
    def _visualize_results(self, frame: np.ndarray, results: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw identification results on the frame
        
        Args:
            frame: Input frame
            results: List of identification results
            
        Returns:
            Frame with visualizations
        """
        output_frame = frame.copy()
        
        for result in results:
            x1, y1, x2, y2 = result['bbox']
            identity = result['identity']
            person_db_id = result['person_db_id']
            similarity = result['face_similarity']
            attention_score = result.get('attention_score')
            
            # Choose color based on identification
            if identity != 'Unknown':
                color = Config.BBOX_COLOR_KNOWN
                label = f"{identity} ({similarity:.2f})"
                if person_db_id:
                    label += f" [{person_db_id}]"
            else:
                color = Config.BBOX_COLOR_UNKNOWN
                label = "Unknown"
            
            # Draw bounding box
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, Config.BBOX_THICKNESS)
            
            # Prepare label text
            label_lines = [label]
            if result['num_faces'] > 1:
                label_lines.append(f"Faces: {result['num_faces']}")
            if attention_score is not None:
                label_lines.append(f"Attention: {attention_score}%")
            # Draw label background and text
            y_offset = y1
            for line in label_lines:
                label_size = cv2.getTextSize(line, Config.TEXT_FONT, 
                                           Config.TEXT_SCALE, Config.TEXT_THICKNESS)[0]
                
                # Draw background rectangle
                cv2.rectangle(output_frame, (x1, y_offset - label_size[1] - 10), 
                             (x1 + label_size[0], y_offset), color, -1)
                
                # Draw text
                cv2.putText(output_frame, line, (x1, y_offset - 5),
                           Config.TEXT_FONT, Config.TEXT_SCALE, 
                           Config.TEXT_COLOR, Config.TEXT_THICKNESS)
                
                y_offset -= (label_size[1] + 15)
        
        # Draw FPS counter
        fps_text = f"FPS: {self.current_fps:.1f}"
        cv2.putText(output_frame, fps_text, (10, 30),
                   Config.TEXT_FONT, Config.TEXT_SCALE, (0, 255, 0), Config.TEXT_THICKNESS)
        
        # Draw database info
        db_stats = self.database.get_database_stats()
        db_text = f"Database: {db_stats.get('total_persons', 0)} persons"
        cv2.putText(output_frame, db_text, (10, 60),
                   Config.TEXT_FONT, Config.TEXT_SCALE, (0, 255, 0), Config.TEXT_THICKNESS)
        
        # Add system info
        system_text = "Organized Database System"
        cv2.putText(output_frame, system_text, (10, 90),
                   Config.TEXT_FONT, 0.5, (255, 255, 0), 1)
        
        return output_frame
    
    def _update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:  # Update every second
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def run_webcam(self, camera_id: int = 0, display: bool = True) -> None:
        """
        Run identification system with webcam input
        
        Args:
            camera_id: Camera device ID
            display: Whether to display the video feed
        """
        try:
            cap = cv2.VideoCapture(camera_id)
            
            if not cap.isOpened():
                logger.error(f"Could not open camera {camera_id}")
                return
            
            # Set camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            logger.info("Starting webcam identification. Press 'q' to quit, 's' to save frame, 'b' to backup database")
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Could not read frame from camera")
                    break
                
                # Process frame
                processed_frame, results = self.process_frame(frame)
                
                # Log results periodically
                if frame_count % 30 == 0 and results:  # Every 30 frames
                    identified_persons = [r for r in results if r['identity'] != 'Unknown']
                    if identified_persons:
                        names = [f"{r['identity']} ({r['person_db_id']})" for r in identified_persons]
                        logger.info(f"Identified: {', '.join(names)}")
                
                if display:
                    cv2.imshow("Organized Person Identification System", processed_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        # Save current frame
                        filename = f"organized_identification_frame_{int(time.time())}.jpg"
                        cv2.imwrite(filename, processed_frame)
                        logger.info(f"Saved frame as {filename}")
                    elif key == ord('b'):
                        # Create database backup
                        backup_name = f"live_backup_{int(time.time())}"
                        if self.database.create_backup(backup_name):
                            logger.info(f"Database backup created: {backup_name}")
                        else:
                            logger.error("Failed to create database backup")
                
                frame_count += 1
            
            cap.release()
            if display:
                cv2.destroyAllWindows()
                
        except Exception as e:
            logger.error(f"Error in webcam mode: {e}")
    
    def run_video_file(self, video_path: str, output_path: str = None, display: bool = True) -> None:
        """
        Run identification system on a video file
        
        Args:
            video_path: Path to input video file
            output_path: Path to save output video (optional)
            display: Whether to display the video feed
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                logger.error(f"Could not open video file: {video_path}")
                return
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"Processing video: {video_path}")
            logger.info(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
            
            # Setup video writer if output path provided
            writer = None
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frame_count = 0
            start_time = time.time()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                processed_frame, results = self.process_frame(frame)
                
                # Write to output video if specified
                if writer:
                    writer.write(processed_frame)
                
                # Display progress
                if frame_count % 30 == 0:  # Every 30 frames
                    progress = (frame_count / total_frames) * 100
                    elapsed = time.time() - start_time
                    eta = (elapsed / frame_count) * (total_frames - frame_count) if frame_count > 0 else 0
                    logger.info(f"Progress: {progress:.1f}% (Frame {frame_count}/{total_frames}, ETA: {eta:.1f}s)")
                
                if display:
                    cv2.imshow("Organized Person Identification System", processed_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                
                frame_count += 1
            
            # Cleanup
            cap.release()
            if writer:
                writer.release()
                logger.info(f"Output video saved: {output_path}")
            
            if display:
                cv2.destroyAllWindows()
            
            total_time = time.time() - start_time
            logger.info(f"Processing completed in {total_time:.2f}s ({frame_count/total_time:.1f} FPS)")
            
        except Exception as e:
            logger.error(f"Error processing video file: {e}")
    
    def get_system_info(self) -> Dict[str, Any]:
        """
        Get system information and statistics
        
        Returns:
            Dictionary with system information
        """
        db_stats = self.database.get_database_stats()
        
        info = {
            'system_name': 'Organized Person Identification System',
            'version': '2.0.0',
            'face_processing': 'OpenCV-based (Simplified)',
            'database_type': 'Organized (Individual directories)',
            'current_fps': self.current_fps,
            'database_stats': db_stats,
            'config': {
                'person_detection_confidence': Config.PERSON_DETECTION_CONFIDENCE,
                'face_recognition_threshold': Config.FACE_RECOGNITION_THRESHOLD,
                'database_path': Config.DATABASE_PATH,
                'device': Config.DEVICE
            }
        }
        
        return info


def main():
    """Command line interface for the organized identification system"""
    parser = argparse.ArgumentParser(description="Organized Person Identification System")
    parser.add_argument('--mode', choices=['webcam', 'video'], default='webcam',
                       help='Input mode: webcam or video file')
    parser.add_argument('--input', type=str, help='Input video file path (for video mode)')
    parser.add_argument('--output', type=str, help='Output video file path (optional)')
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID (default: 0)')
    parser.add_argument('--no-display', action='store_true', help='Disable video display')
    parser.add_argument('--info', action='store_true', help='Show system information')
    
    args = parser.parse_args()
    
    try:
        # Initialize system
        system = SimplePersonIdentificationSystemOrganized()
        
        if args.info:
            # Display system information
            info = system.get_system_info()
            print("\n" + "="*60)
            print("ORGANIZED PERSON IDENTIFICATION SYSTEM INFO")
            print("="*60)
            print(f"System: {info['system_name']} v{info['version']}")
            print(f"Face Processing: {info['face_processing']}")
            print(f"Database Type: {info['database_type']}")
            print(f"Current FPS: {info['current_fps']:.1f}")
            
            db_stats = info['database_stats']
            print(f"Database: {db_stats.get('total_persons', 0)} persons, {db_stats.get('total_embeddings', 0)} embeddings")
            print(f"Database Size: {db_stats.get('total_size_mb', 0)} MB")
            print(f"Database Path: {db_stats.get('database_path', 'Unknown')}")
            print(f"Device: {info['config']['device']}")
            
            # Show individual persons
            if db_stats.get('persons'):
                print(f"\nEnrolled Persons ({len(db_stats['persons'])}):")
                for person_id in db_stats['persons']:
                    metadata = system.database.get_person_metadata(person_id)
                    if metadata:
                        print(f"  • {metadata['name']} ({person_id})")
            
            print("="*60)
            return
        
        # Run system based on mode
        if args.mode == 'webcam':
            system.run_webcam(camera_id=args.camera, display=not args.no_display)
        
        elif args.mode == 'video':
            if not args.input:
                print("Error: --input is required for video mode")
                return
            
            system.run_video_file(args.input, args.output, display=not args.no_display)
        
    except KeyboardInterrupt:
        logger.info("System stopped by user")
    except Exception as e:
        logger.error(f"System error: {e}")
        raise


if __name__ == "__main__":
    main() 