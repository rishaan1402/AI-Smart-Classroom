#!/usr/bin/env python3
"""
Test script for the new organized database system
"""

import os
import sys
import numpy as np
import logging
from typing import List
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class OrganizedDatabaseTester:
    """Test the organized database system"""
    
    def __init__(self):
        self.test_results = {}
        self.errors = []
    
    def test_database_creation(self) -> bool:
        """Test creating a new organized database"""
        print("Testing organized database creation...")
        
        try:
            from face_database_organized import OrganizedFaceDatabase
            
            # Create test database in temporary location
            test_db_path = "test_organized_db"
            db = OrganizedFaceDatabase(test_db_path)
            
            # Check directory structure
            expected_dirs = ["persons", "backups"]
            for dir_name in expected_dirs:
                dir_path = os.path.join(test_db_path, dir_name)
                if not os.path.exists(dir_path):
                    raise Exception(f"Missing directory: {dir_name}")
            
            # Check index file
            index_file = os.path.join(test_db_path, "index.json")
            if not os.path.exists(index_file):
                raise Exception("Missing index.json file")
            
            print("✓ Database structure created successfully")
            
            # Cleanup
            import shutil
            shutil.rmtree(test_db_path, ignore_errors=True)
            
            return True
            
        except Exception as e:
            error_msg = f"Database creation error: {e}"
            print(f"✗ {error_msg}")
            self.errors.append(error_msg)
            return False
    
    def test_person_operations(self) -> bool:
        """Test adding, retrieving, and removing persons"""
        print("\nTesting person operations...")
        
        try:
            from face_database_organized import OrganizedFaceDatabase
            
            # Create test database
            test_db_path = "test_organized_db"
            db = OrganizedFaceDatabase(test_db_path)
            
            # Test data
            test_persons = [
                ("Alice Test", [np.random.rand(512) for _ in range(3)]),
                ("Bob Test", [np.random.rand(512) for _ in range(4)]),
                ("Charlie Test", [np.random.rand(512) for _ in range(2)])
            ]
            
            added_person_ids = []
            
            # Test adding persons
            for name, embeddings in test_persons:
                additional_info = {"test": True, "source": "unit_test"}
                success = db.add_person(name, embeddings, additional_info)
                
                if success:
                    print(f"✓ Added person: {name}")
                    # Find the person ID that was created
                    stats = db.get_database_stats()
                    for person_id in stats["persons"]:
                        metadata = db.get_person_metadata(person_id)
                        if metadata and metadata["name"] == name:
                            added_person_ids.append(person_id)
                            break
                else:
                    raise Exception(f"Failed to add person: {name}")
            
            # Test retrieving persons
            all_persons = db.get_all_persons()
            if len(all_persons) != len(test_persons):
                raise Exception(f"Expected {len(test_persons)} persons, got {len(all_persons)}")
            
            print(f"✓ Retrieved {len(all_persons)} persons")
            
            # Test individual person retrieval
            for person_id in added_person_ids:
                embeddings = db.get_person_embeddings(person_id)
                metadata = db.get_person_metadata(person_id)
                
                if embeddings is None or metadata is None:
                    raise Exception(f"Failed to retrieve data for person: {person_id}")
            
            print("✓ Individual person data retrieval successful")
            
            # Test identification
            if added_person_ids:
                # Use first person's first embedding for identification test
                first_person_id = added_person_ids[0]
                first_person_embeddings = db.get_person_embeddings(first_person_id)
                query_embedding = first_person_embeddings[0]
                
                name, person_id, similarity = db.identify_person(query_embedding, threshold=0.5)
                
                if name and person_id == first_person_id:
                    print(f"✓ Identification successful: {name} (similarity: {similarity:.3f})")
                else:
                    raise Exception("Identification failed")
            
            # Test removing persons
            for person_id in added_person_ids:
                if db.remove_person(person_id):
                    print(f"✓ Removed person: {person_id}")
                else:
                    raise Exception(f"Failed to remove person: {person_id}")
            
            # Verify removal
            final_persons = db.get_all_persons()
            if len(final_persons) != 0:
                raise Exception(f"Expected 0 persons after removal, got {len(final_persons)}")
            
            print("✓ Person removal successful")
            
            # Cleanup
            import shutil
            shutil.rmtree(test_db_path, ignore_errors=True)
            
            return True
            
        except Exception as e:
            error_msg = f"Person operations error: {e}"
            print(f"✗ {error_msg}")
            self.errors.append(error_msg)
            return False
    
    def test_backup_functionality(self) -> bool:
        """Test database backup functionality"""
        print("\nTesting backup functionality...")
        
        try:
            from face_database_organized import OrganizedFaceDatabase
            
            # Create test database with data
            test_db_path = "test_organized_db"
            db = OrganizedFaceDatabase(test_db_path)
            
            # Add test person
            test_embeddings = [np.random.rand(512) for _ in range(2)]
            db.add_person("Test Person", test_embeddings, {"test": True})
            
            # Create backup
            backup_success = db.create_backup("test_backup")
            
            if not backup_success:
                raise Exception("Backup creation failed")
            
            # Check if backup file exists
            backup_file = os.path.join(test_db_path, "backups", "test_backup.zip")
            if not os.path.exists(backup_file):
                raise Exception("Backup file not found")
            
            print("✓ Backup creation successful")
            
            # Cleanup
            import shutil
            shutil.rmtree(test_db_path, ignore_errors=True)
            
            return True
            
        except Exception as e:
            error_msg = f"Backup functionality error: {e}"
            print(f"✗ {error_msg}")
            self.errors.append(error_msg)
            return False
    
    def test_concurrent_access(self) -> bool:
        """Test concurrent database access"""
        print("\nTesting concurrent access...")
        
        try:
            from face_database_organized import OrganizedFaceDatabase
            import threading
            import time
            
            # Create test database
            test_db_path = "test_organized_db"
            db = OrganizedFaceDatabase(test_db_path)
            
            # Test concurrent operations
            results = []
            
            def add_person_thread(person_name, thread_id):
                try:
                    embeddings = [np.random.rand(512) for _ in range(2)]
                    success = db.add_person(f"{person_name}_{thread_id}", embeddings, {"thread_id": thread_id})
                    results.append(success)
                except Exception as e:
                    results.append(False)
            
            # Create multiple threads
            threads = []
            for i in range(3):
                thread = threading.Thread(target=add_person_thread, args=("ConcurrentTest", i))
                threads.append(thread)
            
            # Start all threads
            for thread in threads:
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join()
            
            # Check results
            if len(results) != 3 or not all(results):
                raise Exception("Concurrent operations failed")
            
            # Verify all persons were added
            persons = db.get_all_persons()
            concurrent_persons = [p for p in persons if "ConcurrentTest" in p["name"]]
            
            if len(concurrent_persons) != 3:
                raise Exception(f"Expected 3 concurrent persons, got {len(concurrent_persons)}")
            
            print("✓ Concurrent access successful")
            
            # Cleanup
            for person in concurrent_persons:
                db.remove_person(person["person_id"])
            
            import shutil
            shutil.rmtree(test_db_path, ignore_errors=True)
            
            return True
            
        except Exception as e:
            error_msg = f"Concurrent access error: {e}"
            print(f"✗ {error_msg}")
            self.errors.append(error_msg)
            return False
    
    def run_all_tests(self) -> bool:
        """Run all organized database tests"""
        print("=" * 60)
        print("ORGANIZED DATABASE SYSTEM TESTS")
        print("=" * 60)
        
        tests = [
            ("Database Creation", self.test_database_creation),
            ("Person Operations", self.test_person_operations),
            ("Backup Functionality", self.test_backup_functionality),
            ("Concurrent Access", self.test_concurrent_access),
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                self.test_results[test_name] = result
                if result:
                    passed += 1
            except Exception as e:
                print(f"✗ {test_name} test crashed: {e}")
                self.test_results[test_name] = False
                self.errors.append(f"{test_name}: {e}")
        
        # Print summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print(f"Tests passed: {passed}/{total}")
        
        if self.errors:
            print("\nErrors encountered:")
            for error in self.errors:
                print(f"  • {error}")
        
        if passed == total:
            print("\n🎉 All tests passed! Organized database system is working perfectly.")
            print("\nThe new organized database provides:")
            print("  • Individual directories for each person")
            print("  • Better organization and scalability")
            print("  • Backup functionality")
            print("  • Migration from old database format")
        else:
            print(f"\n⚠️  {total - passed} test(s) failed.")
        
        return passed == total

def main():
    """Main test function"""
    tester = OrganizedDatabaseTester()
    return tester.run_all_tests()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 