#!/usr/bin/env python3
"""
Simple Enrollment System with Organized Database
Uses the new organized database structure for better scalability
"""

import os
import cv2
import numpy as np
from typing import List, Optional, Dict, Any
import logging
from pathlib import Path
import argparse
from config import Config
from face_processor_simple import SimpleFaceProcessor
from face_database_organized import OrganizedFaceDatabase

logger = logging.getLogger(__name__)

class SimpleEnrollmentSystemOrganized:
    """
    Simple enrollment system using organized database structure
    """
    
    def __init__(self):
        """Initialize enrollment system"""
        self.face_processor = SimpleFaceProcessor()
        self.database = OrganizedFaceDatabase()
        
        # Ensure database directory exists
        Config.create_directories()
    
    def enroll_from_images(self, person_name: str, image_paths: List[str], 
                          additional_info: Dict[str, Any] = None) -> bool:
        """
        Enroll a person using multiple images
        
        Args:
            person_name: Unique name for the person
            image_paths: List of image file paths
            additional_info: Additional metadata
            
        Returns:
            True if enrollment successful, False otherwise
        """
        try:
            if len(image_paths) < Config.MIN_ENROLLMENT_IMAGES:
                logger.error(f"Need at least {Config.MIN_ENROLLMENT_IMAGES} images for enrollment")
                return False
            
            # Limit number of images
            if len(image_paths) > Config.MAX_ENROLLMENT_IMAGES:
                image_paths = image_paths[:Config.MAX_ENROLLMENT_IMAGES]
                logger.warning(f"Limited to {Config.MAX_ENROLLMENT_IMAGES} images for {person_name}")
            
            # Process each image and extract embeddings
            embeddings = []
            reference_images = []
            processed_images = 0
            
            for img_path in image_paths:
                try:
                    # Load image
                    image = cv2.imread(img_path)
                    if image is None:
                        logger.warning(f"Could not load image: {img_path}")
                        continue
                    
                    # Extract faces and embeddings
                    faces = self.face_processor.extract_faces_from_person(image)
                    
                    if not faces:
                        logger.warning(f"No faces found in image: {img_path}")
                        continue
                    
                    # Use the largest face (assuming it's the main subject)
                    largest_face = max(faces, key=lambda x: x[0].shape[0] * x[0].shape[1])
                    face_img, _ = largest_face
                    
                    # Extract embedding
                    embedding = self.face_processor.extract_face_embedding(face_img)
                    
                    if embedding is not None:
                        embeddings.append(embedding)
                        reference_images.append(face_img)  # Store the face image
                        processed_images += 1
                        logger.info(f"Processed image {processed_images}: {os.path.basename(img_path)}")
                    else:
                        logger.warning(f"Could not extract embedding from: {img_path}")
                        
                except Exception as e:
                    logger.error(f"Error processing image {img_path}: {e}")
                    continue
            
            # Check if we have enough embeddings
            if len(embeddings) < Config.MIN_ENROLLMENT_IMAGES:
                logger.error(f"Only extracted {len(embeddings)} valid embeddings, need at least {Config.MIN_ENROLLMENT_IMAGES}")
                return False
            
            # Prepare enrollment info
            enrollment_info = {
                'enrollment_method': 'images',
                'source_images': [os.path.basename(path) for path in image_paths[:len(embeddings)]],
                'processed_images': processed_images
            }
            
            if additional_info:
                enrollment_info.update(additional_info)
            
            # Add to organized database (with reference images)
            success = self.database.add_person(person_name, embeddings, enrollment_info, reference_images)
            
            if success:
                logger.info(f"Successfully enrolled '{person_name}' with {len(embeddings)} embeddings")
            else:
                logger.error(f"Failed to enroll '{person_name}'")
            
            return success
            
        except Exception as e:
            logger.error(f"Error during enrollment: {e}")
            return False
    
    def enroll_from_folder(self, person_name: str, folder_path: str,
                          additional_info: Dict[str, Any] = None) -> bool:
        """
        Enroll a person using all images from a folder
        
        Args:
            person_name: Unique name for the person
            folder_path: Path to folder containing images
            additional_info: Additional metadata
            
        Returns:
            True if enrollment successful, False otherwise
        """
        try:
            folder_path = Path(folder_path)
            
            if not folder_path.exists() or not folder_path.is_dir():
                logger.error(f"Folder does not exist: {folder_path}")
                return False
            
            # Find all image files
            image_paths = []
            for ext in Config.ENROLLMENT_IMAGE_FORMATS:
                image_paths.extend(folder_path.glob(f"*{ext}"))
                image_paths.extend(folder_path.glob(f"*{ext.upper()}"))
            
            if not image_paths:
                logger.error(f"No image files found in folder: {folder_path}")
                return False
            
            # Convert to strings and sort
            image_paths = sorted([str(path) for path in image_paths])
            
            logger.info(f"Found {len(image_paths)} images in folder for {person_name}")
            
            # Add folder info to metadata
            folder_info = {'source_folder': str(folder_path)}
            if additional_info:
                folder_info.update(additional_info)
            
            return self.enroll_from_images(person_name, image_paths, folder_info)
            
        except Exception as e:
            logger.error(f"Error enrolling from folder: {e}")
            return False
    
    def enroll_from_webcam(self, person_name: str, num_captures: int = None,
                          additional_info: Dict[str, Any] = None) -> bool:
        """
        Enroll a person by capturing images from webcam
        
        Args:
            person_name: Unique name for the person
            num_captures: Number of images to capture
            additional_info: Additional metadata
            
        Returns:
            True if enrollment successful, False otherwise
        """
        try:
            num_captures = num_captures or Config.MIN_ENROLLMENT_IMAGES
            
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                logger.error("Could not open webcam")
                return False
            
            captured_embeddings = []
            captured_faces = []
            capture_count = 0
            
            print(f"\nEnrolling '{person_name}' - Press SPACE to capture, 'q' to quit")
            print(f"Need to capture {num_captures} good face images")
            
            while capture_count < num_captures:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Could not read frame from webcam")
                    break
                
                # Detect faces in current frame
                faces = self.face_processor.extract_faces_from_person(frame)
                
                # Draw face boxes for feedback
                display_frame = frame.copy()
                for face_img, detection in faces:
                    bbox = detection['bbox']
                    cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                
                # Display instructions
                cv2.putText(display_frame, f"Captured: {capture_count}/{num_captures}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display_frame, "Press SPACE to capture, 'q' to quit", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                if faces:
                    cv2.putText(display_frame, f"Faces detected: {len(faces)}", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(display_frame, "No faces detected", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                cv2.imshow("Organized Enrollment - Face Capture", display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord(' '):  # Space to capture
                    if faces:
                        # Use the largest face
                        largest_face = max(faces, key=lambda x: x[0].shape[0] * x[0].shape[1])
                        face_img, _ = largest_face
                        
                        # Extract embedding
                        embedding = self.face_processor.extract_face_embedding(face_img)
                        
                        if embedding is not None:
                            captured_embeddings.append(embedding)
                            captured_faces.append(face_img)
                            capture_count += 1
                            print(f"Captured image {capture_count}/{num_captures}")
                        else:
                            print("Failed to extract embedding, try again")
                    else:
                        print("No faces detected, position yourself in front of camera")
                
                elif key == ord('q'):  # Quit
                    print("Enrollment cancelled")
                    cap.release()
                    cv2.destroyAllWindows()
                    return False
            
            cap.release()
            cv2.destroyAllWindows()
            
            if len(captured_embeddings) >= Config.MIN_ENROLLMENT_IMAGES:
                # Add to organized database
                webcam_info = {
                    'enrollment_method': 'webcam',
                    'captured_images': len(captured_embeddings)
                }
                
                if additional_info:
                    webcam_info.update(additional_info)
                
                # Include captured face images
                success = self.database.add_person(person_name, captured_embeddings, webcam_info, captured_faces)
                
                if success:
                    print(f"Successfully enrolled '{person_name}' with {len(captured_embeddings)} face captures")
                else:
                    print(f"Failed to save '{person_name}' to database")
                
                return success
            else:
                print(f"Not enough face captures ({len(captured_embeddings)}), enrollment failed")
                return False
                
        except Exception as e:
            logger.error(f"Error during webcam enrollment: {e}")
            return False
    
    def list_enrolled_persons(self) -> None:
        """List all enrolled persons with their information"""
        try:
            persons = self.database.get_all_persons()
            
            if not persons:
                print("No persons enrolled in the database")
                return
            
            print(f"\nEnrolled Persons ({len(persons)}):")
            print("-" * 70)
            
            for person in persons:
                print(f"Name: {person['name']}")
                print(f"  Person ID: {person['person_id']}")
                print(f"  Embeddings: {person['num_embeddings']}")
                print(f"  Added: {person['date_added']}")
                
                metadata = person.get('metadata', {})
                print(f"  Method: {metadata.get('enrollment_method', 'Unknown')}")
                
                if metadata.get('has_reference_images'):
                    print(f"  Reference Images: {metadata.get('num_reference_images', 0)}")
                
                if metadata.get('migrated_from_old_db'):
                    print(f"  Migrated: Yes ({metadata.get('migration_date', 'Unknown date')})")
                
                print()
                    
        except Exception as e:
            logger.error(f"Error listing persons: {e}")
    
    def remove_person(self, person_name: str) -> bool:
        """
        Remove a person from the database by name
        
        Args:
            person_name: Name of person to remove
            
        Returns:
            True if removed successfully, False otherwise
        """
        try:
            # Find person ID by name
            persons = self.database.get_all_persons()
            person_id = None
            
            for person in persons:
                if person['name'].lower() == person_name.lower():
                    person_id = person['person_id']
                    break
            
            if person_id:
                return self.database.remove_person(person_id)
            else:
                print(f"Person '{person_name}' not found in database")
                return False
                
        except Exception as e:
            logger.error(f"Error removing person: {e}")
            return False
    
    def create_backup(self, backup_name: str = None) -> bool:
        """
        Create a backup of the database
        
        Args:
            backup_name: Optional backup name
            
        Returns:
            True if backup created successfully
        """
        return self.database.create_backup(backup_name)
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        return self.database.get_database_stats()


def main():
    """Command line interface for organized enrollment system"""
    parser = argparse.ArgumentParser(description="Organized Person Enrollment System")
    parser.add_argument('command', choices=['enroll', 'list', 'remove', 'backup', 'stats'], 
                       help='Command to execute')
    parser.add_argument('--name', type=str, help='Person name for enrollment/removal')
    parser.add_argument('--images', nargs='+', help='List of image paths for enrollment')
    parser.add_argument('--folder', type=str, help='Folder path containing images')
    parser.add_argument('--webcam', action='store_true', help='Use webcam for enrollment')
    parser.add_argument('--captures', type=int, default=4, 
                       help='Number of webcam captures (default: 4)')
    parser.add_argument('--backup-name', type=str, help='Custom backup name')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Initialize enrollment system
    enrollment = SimpleEnrollmentSystemOrganized()
    
    if args.command == 'enroll':
        if not args.name:
            print("Error: --name is required for enrollment")
            return
        
        if args.webcam:
            success = enrollment.enroll_from_webcam(args.name, args.captures)
        elif args.folder:
            success = enrollment.enroll_from_folder(args.name, args.folder)
        elif args.images:
            success = enrollment.enroll_from_images(args.name, args.images)
        else:
            print("Error: Must specify --webcam, --folder, or --images for enrollment")
            return
        
        if success:
            print(f"Successfully enrolled '{args.name}'")
        else:
            print(f"Failed to enroll '{args.name}'")
    
    elif args.command == 'list':
        enrollment.list_enrolled_persons()
    
    elif args.command == 'remove':
        if not args.name:
            print("Error: --name is required for removal")
            return
        
        success = enrollment.remove_person(args.name)
        if success:
            print(f"Successfully removed '{args.name}'")
        else:
            print(f"Failed to remove '{args.name}'")
    
    elif args.command == 'backup':
        success = enrollment.create_backup(args.backup_name)
        if success:
            print("Backup created successfully")
        else:
            print("Failed to create backup")
    
    elif args.command == 'stats':
        stats = enrollment.get_database_stats()
        print("\nDatabase Statistics:")
        print("-" * 40)
        print(f"Total persons: {stats.get('total_persons', 0)}")
        print(f"Total embeddings: {stats.get('total_embeddings', 0)}")
        print(f"Database size: {stats.get('total_size_mb', 0)} MB")
        print(f"Location: {stats.get('database_path', 'Unknown')}")
        print(f"Created: {stats.get('created', 'Unknown')}")
        print(f"Last updated: {stats.get('last_updated', 'Unknown')}")


if __name__ == "__main__":
    main() 