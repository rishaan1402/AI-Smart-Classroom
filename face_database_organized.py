"""
Organized Face Database Manager
Improved storage system with individual person directories for better organization and scalability
"""

import os
import json
import numpy as np
import shutil
import zipfile
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
from pathlib import Path
from config import Config
import uuid

logger = logging.getLogger(__name__)

class OrganizedFaceDatabase:
    """
    Organized face database with individual person directories
    """
    
    def __init__(self, database_path: str = None):
        """
        Initialize organized face database
        
        Args:
            database_path: Path to database directory
        """
        self.database_path = Path(database_path or Config.DATABASE_PATH)
        self.persons_dir = self.database_path / "persons"
        self.backups_dir = self.database_path / "backups"
        self.index_file = self.database_path / "index.json"
        
        # Create directory structure
        self._create_directory_structure()
        
        # Load master index
        self.index = self._load_index()
        
        logger.info(f"Organized database initialized at {self.database_path}")
    
    def _create_directory_structure(self):
        """Create the organized directory structure"""
        self.database_path.mkdir(exist_ok=True)
        self.persons_dir.mkdir(exist_ok=True)
        self.backups_dir.mkdir(exist_ok=True)
        
        # Create index file if it doesn't exist
        if not self.index_file.exists():
            empty_index = self._create_empty_index()
            with open(self.index_file, 'w') as f:
                json.dump(empty_index, f, indent=2, default=str)
    
    def _load_index(self) -> Dict[str, Any]:
        """
        Load the master index
        
        Returns:
            Master index dictionary
        """
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading index: {e}")
                return self._create_empty_index()
        else:
            return self._create_empty_index()
    
    def _create_empty_index(self) -> Dict[str, Any]:
        """Create empty index structure"""
        return {
            "version": "1.0",
            "created": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),
            "total_persons": 0,
            "persons": {}
        }
    
    def _save_index(self) -> bool:
        """
        Save the master index
        
        Returns:
            True if saved successfully
        """
        try:
            self.index["last_updated"] = datetime.now().isoformat()
            with open(self.index_file, 'w') as f:
                json.dump(self.index, f, indent=2, default=str)
            return True
        except Exception as e:
            logger.error(f"Error saving index: {e}")
            return False
    
    def _generate_person_id(self, name: str) -> str:
        """
        Generate unique person ID
        
        Args:
            name: Person name
            
        Returns:
            Unique person ID
        """
        # Clean name for filesystem
        clean_name = "".join(c.lower() if c.isalnum() else "_" for c in name)[:20]
        
        # Find next available ID
        counter = 1
        while True:
            person_id = f"{clean_name}_{counter:03d}"
            if person_id not in self.index["persons"]:
                return person_id
            counter += 1
    
    def _get_person_directory(self, person_id: str) -> Path:
        """Get person directory path"""
        return self.persons_dir / person_id
    
    def add_person(self, person_name: str, embeddings: List[np.ndarray], 
                   additional_info: Dict[str, Any] = None, 
                   reference_images: List[np.ndarray] = None) -> bool:
        """
        Add a new person to the organized database
        
        Args:
            person_name: Person name
            embeddings: List of face embeddings
            additional_info: Additional metadata
            reference_images: Optional reference face images
            
        Returns:
            True if added successfully
        """
        try:
            if not embeddings:
                logger.error("No embeddings provided")
                return False
            
            # Generate unique person ID
            person_id = self._generate_person_id(person_name)
            person_dir = self._get_person_directory(person_id)
            
            # Create person directory
            person_dir.mkdir(exist_ok=True)
            
            # Create images subdirectory if reference images provided
            if reference_images:
                images_dir = person_dir / "images"
                images_dir.mkdir(exist_ok=True)
            
            # Prepare metadata
            metadata = {
                "person_id": person_id,
                "name": person_name,
                "num_embeddings": len(embeddings),
                "date_added": datetime.now().isoformat(),
                "last_updated": datetime.now().isoformat(),
                "embedding_shape": list(embeddings[0].shape) if embeddings else None,
                "has_reference_images": reference_images is not None,
                "num_reference_images": len(reference_images) if reference_images else 0
            }
            
            if additional_info:
                metadata.update(additional_info)
            
            # Save embeddings as NumPy array
            embeddings_array = np.array(embeddings)
            embeddings_file = person_dir / "embeddings.npy"
            np.save(embeddings_file, embeddings_array)
            
            # Save metadata
            metadata_file = person_dir / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            # Save reference images if provided
            if reference_images:
                images_dir = person_dir / "images"
                for i, img in enumerate(reference_images):
                    img_file = images_dir / f"face_{i+1:03d}.jpg"
                    # Convert numpy array to image and save
                    import cv2
                    cv2.imwrite(str(img_file), img)
            
            # Update master index
            self.index["persons"][person_id] = {
                "name": person_name,
                "person_id": person_id,
                "date_added": metadata["date_added"],
                "num_embeddings": len(embeddings),
                "directory": str(person_dir.relative_to(self.database_path))
            }
            self.index["total_persons"] = len(self.index["persons"])
            
            # Save index
            if self._save_index():
                logger.info(f"Added person '{person_name}' with ID '{person_id}' and {len(embeddings)} embeddings")
                return True
            else:
                # Rollback on index save failure
                shutil.rmtree(person_dir, ignore_errors=True)
                if person_id in self.index["persons"]:
                    del self.index["persons"][person_id]
                return False
                
        except Exception as e:
            logger.error(f"Error adding person {person_name}: {e}")
            return False
    
    def get_person_embeddings(self, person_id: str) -> Optional[np.ndarray]:
        """
        Get embeddings for a specific person
        
        Args:
            person_id: Person ID
            
        Returns:
            NumPy array of embeddings or None
        """
        try:
            if person_id not in self.index["persons"]:
                return None
            
            person_dir = self._get_person_directory(person_id)
            embeddings_file = person_dir / "embeddings.npy"
            
            if embeddings_file.exists():
                return np.load(embeddings_file)
            else:
                logger.warning(f"Embeddings file not found for person {person_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading embeddings for {person_id}: {e}")
            return None
    
    def get_person_metadata(self, person_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a specific person
        
        Args:
            person_id: Person ID
            
        Returns:
            Metadata dictionary or None
        """
        try:
            if person_id not in self.index["persons"]:
                return None
            
            person_dir = self._get_person_directory(person_id)
            metadata_file = person_dir / "metadata.json"
            
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    return json.load(f)
            else:
                logger.warning(f"Metadata file not found for person {person_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading metadata for {person_id}: {e}")
            return None
    
    def identify_person(self, query_embedding: np.ndarray, 
                       threshold: float = None) -> Tuple[Optional[str], Optional[str], float]:
        """
        Identify a person based on face embedding
        
        Args:
            query_embedding: Face embedding to identify
            threshold: Similarity threshold
            
        Returns:
            (person_name, person_id, best_similarity) or (None, None, 0.0)
        """
        try:
            threshold = threshold or Config.FACE_RECOGNITION_THRESHOLD
            best_match_name = None
            best_match_id = None
            best_similarity = 0.0
            
            for person_id in self.index["persons"]:
                person_embeddings = self.get_person_embeddings(person_id)
                
                if person_embeddings is not None:
                    # Compare with all embeddings for this person
                    similarities = []
                    
                    for embedding in person_embeddings:
                        similarity = self._calculate_similarity(query_embedding, embedding)
                        similarities.append(similarity)
                    
                    # Use the best similarity for this person
                    if similarities:
                        person_best_sim = max(similarities)
                        
                        if person_best_sim > best_similarity and person_best_sim >= threshold:
                            best_similarity = person_best_sim
                            best_match_id = person_id
                            best_match_name = self.index["persons"][person_id]["name"]
            
            return best_match_name, best_match_id, best_similarity
            
        except Exception as e:
            logger.error(f"Error identifying person: {e}")
            return None, None, 0.0
    
    def _calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings"""
        try:
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            cosine_sim = np.dot(embedding1, embedding2) / (norm1 * norm2)
            similarity = (cosine_sim + 1) / 2
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def remove_person(self, person_id: str) -> bool:
        """
        Remove a person from the database
        
        Args:
            person_id: Person ID to remove
            
        Returns:
            True if removed successfully
        """
        try:
            if person_id not in self.index["persons"]:
                logger.error(f"Person ID '{person_id}' not found in database")
                return False
            
            person_dir = self._get_person_directory(person_id)
            
            # Remove person directory
            if person_dir.exists():
                shutil.rmtree(person_dir)
            
            # Remove from index
            person_name = self.index["persons"][person_id]["name"]
            del self.index["persons"][person_id]
            self.index["total_persons"] = len(self.index["persons"])
            
            # Save index
            if self._save_index():
                logger.info(f"Removed person '{person_name}' (ID: {person_id}) from database")
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Error removing person {person_id}: {e}")
            return False
    
    def get_all_persons(self) -> List[Dict[str, Any]]:
        """
        Get list of all persons in database
        
        Returns:
            List of person information dictionaries
        """
        try:
            persons = []
            for person_id, person_info in self.index["persons"].items():
                metadata = self.get_person_metadata(person_id)
                if metadata:
                    persons.append({
                        "person_id": person_id,
                        "name": person_info["name"],
                        "num_embeddings": person_info["num_embeddings"],
                        "date_added": person_info["date_added"],
                        "metadata": metadata
                    })
            return persons
        except Exception as e:
            logger.error(f"Error getting all persons: {e}")
            return []
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics"""
        try:
            total_embeddings = sum(person["num_embeddings"] for person in self.index["persons"].values())
            
            # Calculate directory sizes
            total_size = 0
            for person_id in self.index["persons"]:
                person_dir = self._get_person_directory(person_id)
                if person_dir.exists():
                    for file_path in person_dir.rglob("*"):
                        if file_path.is_file():
                            total_size += file_path.stat().st_size
            
            stats = {
                "version": self.index.get("version", "1.0"),
                "total_persons": self.index["total_persons"],
                "total_embeddings": total_embeddings,
                "avg_embeddings_per_person": total_embeddings / self.index["total_persons"] if self.index["total_persons"] > 0 else 0,
                "database_path": str(self.database_path),
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "created": self.index.get("created"),
                "last_updated": self.index.get("last_updated"),
                "persons": list(self.index["persons"].keys())
            }
            
            return stats
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}
    
    def create_backup(self, backup_name: str = None) -> bool:
        """
        Create a backup of the entire database
        
        Args:
            backup_name: Optional backup name
            
        Returns:
            True if backup created successfully
        """
        try:
            if backup_name is None:
                backup_name = f"backup_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
            
            backup_file = self.backups_dir / f"{backup_name}.zip"
            
            with zipfile.ZipFile(backup_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add index file
                zipf.write(self.index_file, "index.json")
                
                # Add all person directories
                for person_id in self.index["persons"]:
                    person_dir = self._get_person_directory(person_id)
                    if person_dir.exists():
                        for file_path in person_dir.rglob("*"):
                            if file_path.is_file():
                                arc_path = file_path.relative_to(self.database_path)
                                zipf.write(file_path, arc_path)
            
            logger.info(f"Database backup created: {backup_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return False
    
    def migrate_from_old_database(self, old_database_path: str) -> bool:
        """
        Migrate data from old pickle-based database
        
        Args:
            old_database_path: Path to old database directory
            
        Returns:
            True if migration successful
        """
        try:
            import pickle
            
            old_embeddings_file = os.path.join(old_database_path, "embeddings.pkl")
            old_metadata_file = os.path.join(old_database_path, "metadata.json")
            
            if not (os.path.exists(old_embeddings_file) and os.path.exists(old_metadata_file)):
                logger.error("Old database files not found")
                return False
            
            # Load old data
            with open(old_embeddings_file, 'rb') as f:
                old_embeddings = pickle.load(f)
            
            with open(old_metadata_file, 'r') as f:
                old_metadata = json.load(f)
            
            # Migrate each person
            migrated_count = 0
            for person_name, embeddings in old_embeddings.items():
                additional_info = old_metadata.get(person_name, {})
                additional_info["migrated_from_old_db"] = True
                additional_info["migration_date"] = datetime.now().isoformat()
                
                if self.add_person(person_name, embeddings, additional_info):
                    migrated_count += 1
                else:
                    logger.warning(f"Failed to migrate person: {person_name}")
            
            logger.info(f"Migration completed: {migrated_count} persons migrated")
            return migrated_count > 0
            
        except Exception as e:
            logger.error(f"Error during migration: {e}")
            return False 