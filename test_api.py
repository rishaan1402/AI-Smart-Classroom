#!/usr/bin/env python3
"""
API Server Test Script
Test the Person Identification API endpoints
"""

import requests
import json
import time
import base64
from pathlib import Path

class APITester:
    """Test the API server functionality"""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_health_check(self):
        """Test health check endpoint"""
        print("🏥 Testing health check...")
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Health check passed: {data['status']}")
                print(f"   Components: {data['components']}")
                print(f"   Database: {data['database_stats']}")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def test_system_info(self):
        """Test system info endpoint"""
        print("\n📊 Testing system info...")
        try:
            response = self.session.get(f"{self.base_url}/system/info")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ System info retrieved:")
                print(f"   System: {data['system_name']} v{data['version']}")
                print(f"   Database: {data['database_stats']['total_persons']} persons")
                print(f"   Embeddings: {data['database_stats']['total_embeddings']}")
                return True
            else:
                print(f"❌ System info failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ System info error: {e}")
            return False
    
    def test_get_persons(self):
        """Test get persons endpoint"""
        print("\n👥 Testing get persons...")
        try:
            response = self.session.get(f"{self.base_url}/persons")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Retrieved {len(data)} persons:")
                for person in data:
                    print(f"   • {person['name']} ({person['person_id']})")
                return data
            else:
                print(f"❌ Get persons failed: {response.status_code}")
                return []
        except Exception as e:
            print(f"❌ Get persons error: {e}")
            return []
    
    def test_database_stats(self):
        """Test database stats endpoint"""
        print("\n📈 Testing database stats...")
        try:
            response = self.session.get(f"{self.base_url}/database/stats")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Database stats retrieved:")
                print(f"   Total persons: {data['total_persons']}")
                print(f"   Total embeddings: {data['total_embeddings']}")
                print(f"   Database size: {data['total_size_mb']} MB")
                return True
            else:
                print(f"❌ Database stats failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Database stats error: {e}")
            return False
    
    def test_enroll_images(self, demo_person="Alice"):
        """Test enrollment from images"""
        print(f"\n➕ Testing enrollment for {demo_person}...")
        
        # Check if demo faces exist
        demo_dir = Path("demo_faces") / demo_person
        if not demo_dir.exists():
            print(f"❌ Demo faces not found for {demo_person}")
            print("   Run: python3 create_demo_faces.py")
            return False
        
        # Get image files
        image_files = list(demo_dir.glob("*.jpg"))[:3]  # Use first 3 images
        
        if not image_files:
            print(f"❌ No image files found in {demo_dir}")
            return False
        
        try:
            # Prepare files for upload
            files = []
            for img_path in image_files:
                with open(img_path, 'rb') as f:
                    files.append(('images', (img_path.name, f.read(), 'image/jpeg')))
            
            # Prepare data
            data = {
                'name': f"API_{demo_person}",
                'additional_info': json.dumps({
                    'test_enrollment': True,
                    'source': 'api_test'
                })
            }
            
            response = self.session.post(
                f"{self.base_url}/enroll/images",
                files=files,
                data=data
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Enrollment successful: {result['message']}")
                return True
            else:
                print(f"❌ Enrollment failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Enrollment error: {e}")
            return False
    
    def test_identify_image(self, demo_person="Alice"):
        """Test identification from image"""
        print(f"\n🔍 Testing identification for {demo_person}...")
        
        # Get a test image
        demo_dir = Path("demo_faces") / demo_person
        if not demo_dir.exists():
            print(f"❌ Demo faces not found for {demo_person}")
            return False
        
        image_files = list(demo_dir.glob("*.jpg"))
        if not image_files:
            print(f"❌ No image files found in {demo_dir}")
            return False
        
        test_image = image_files[0]  # Use first image
        
        try:
            with open(test_image, 'rb') as f:
                files = {'image': (test_image.name, f.read(), 'image/jpeg')}
            
            response = self.session.post(
                f"{self.base_url}/identify/image",
                files=files
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Identification successful:")
                print(f"   Persons detected: {result['num_persons_detected']}")
                print(f"   Identified: {result['identified_persons']}")
                
                for detection in result['detections']:
                    print(f"   • {detection['identity']} (confidence: {detection['face_similarity']:.3f})")
                
                return True
            else:
                print(f"❌ Identification failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Identification error: {e}")
            return False
    
    def test_remove_person(self, person_name):
        """Test person removal"""
        print(f"\n🗑️ Testing removal of {person_name}...")
        try:
            response = self.session.delete(f"{self.base_url}/persons/{person_name}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Removal successful: {result['message']}")
                return True
            else:
                print(f"❌ Removal failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Removal error: {e}")
            return False
    
    def test_backup(self):
        """Test database backup"""
        print("\n💾 Testing database backup...")
        try:
            response = self.session.post(f"{self.base_url}/database/backup")
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Backup successful: {result['message']}")
                return True
            else:
                print(f"❌ Backup failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Backup error: {e}")
            return False
    
    def run_all_tests(self):
        """Run all API tests"""
        print("🚀 Starting API Server Tests")
        print("=" * 50)
        
        # Wait for server to be ready
        print("⏳ Waiting for server to be ready...")
        max_retries = 10
        for i in range(max_retries):
            try:
                response = self.session.get(f"{self.base_url}/health", timeout=5)
                if response.status_code == 200:
                    break
            except:
                pass
            time.sleep(1)
            print(f"   Retry {i+1}/{max_retries}...")
        else:
            print("❌ Server not responding. Make sure it's running:")
            print("   python3 api_server.py")
            return False
        
        tests = [
            ("Health Check", self.test_health_check),
            ("System Info", self.test_system_info),
            ("Get Persons", self.test_get_persons),
            ("Database Stats", self.test_database_stats),
            ("Enroll Images", lambda: self.test_enroll_images("Bob")),
            ("Identify Image", lambda: self.test_identify_image("Bob")),
            ("Database Backup", self.test_backup),
        ]
        
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            try:
                result = test_func()
                if result:
                    passed += 1
                time.sleep(0.5)  # Small delay between tests
            except Exception as e:
                print(f"❌ {test_name} test crashed: {e}")
        
        # Cleanup - remove test person
        try:
            self.test_remove_person("API_Bob")
        except:
            pass
        
        print("\n" + "=" * 50)
        print(f"🏁 Test Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All API tests passed! Your API server is working perfectly.")
        else:
            print(f"⚠️  {total - passed} test(s) failed.")
        
        return passed == total

def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Person Identification API")
    parser.add_argument("--url", default="http://localhost:8000", 
                       help="API server URL (default: http://localhost:8000)")
    
    args = parser.parse_args()
    
    tester = APITester(args.url)
    success = tester.run_all_tests()
    
    if success:
        print("\n🌐 Web Interface Available:")
        print(f"   {args.url}")
        print("\n📚 API Documentation:")
        print(f"   {args.url}/docs")
        print(f"   {args.url}/redoc")
    
    return success

if __name__ == "__main__":
    main() 