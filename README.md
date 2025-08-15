# Person Detection and Identification System

A real-time person detection and identification system using YOLOv8 for person detection and OpenCV-based face processing with an organized database structure.

## 🎯 **System Overview**

This system provides:
- **Real-time person detection** using YOLOv8
- **Face detection and recognition** using OpenCV
- **Organized database** with individual person directories
- **Webcam and video file support**
- **Easy enrollment** from images, folders, or webcam
- **Automatic backup system**
- **Scalable architecture** for production use

## 🏗️ **System Architecture**

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Video Input   │───▶│  Person Detector │───▶│  Face Processor │
│ (Webcam/Video)  │    │    (YOLOv8)      │    │   (OpenCV)      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                          │
┌─────────────────┐    ┌──────────────────┐              │
│  Visualization  │◀───│   Identification │◀─────────────┘
│   & Display     │    │     System       │
└─────────────────┘    └──────────────────┘
                                │
                       ┌──────────────────┐
                       │ Organized Database│
                       │  (Individual     │
                       │  Directories)    │
                       └──────────────────┘
```

## 📁 **Database Structure**

The system uses an organized database with individual directories for each person:

```
face_database/
├── organized/
│   ├── index.json              # Master index
│   ├── persons/                # Individual person directories
│   │   ├── alice_001/         # Unique person ID
│   │   │   ├── metadata.json  # Person metadata
│   │   │   ├── embeddings.npy # Face embeddings
│   │   │   └── images/        # Reference images
│   │   ├── bob_001/
│   │   └── charlie_002/
│   └── backups/               # Automatic backups
│       └── backup_2025-08-07.zip
└── backup_before_migration/    # Legacy backup
```

### **Key Benefits:**
- ✅ **Individual person directories** - Clean organization
- ✅ **Unique person IDs** - Prevents naming conflicts
- ✅ **Fast access** - Load only required data
- ✅ **Scalable** - Supports thousands of persons
- ✅ **Reference images** - Store original face images
- ✅ **Automatic backups** - Built-in data protection

## 🚀 **Quick Start**

### **1. Installation**

```bash
# Clone or navigate to the project directory
cd "Person Detection and Identification Model"

# Install dependencies
pip3 install -r requirements.txt
```

### **2. Quick Demo**

```bash
# Create demo faces for testing
python3 create_demo_faces.py

# Enroll demo persons
python3 simple_enrollment_organized.py enroll --name "Alice" --folder demo_faces/Alice/
python3 simple_enrollment_organized.py enroll --name "Bob" --folder demo_faces/Bob/

# List enrolled persons
python3 simple_enrollment_organized.py list

# Run real-time identification
python3 simple_identification_organized.py
```

### **3. System Information**

```bash
# Show detailed system info
python3 simple_identification_organized.py --info

# Show database statistics
python3 simple_enrollment_organized.py stats
```

## 🌐 **Web API Server**

### **Start the API Server:**
```bash
python3 api_server.py
# Server starts at http://localhost:8000
```

### **Web Interface Features:**
- **📹 Live Video Stream** - Real-time person identification
- **👥 Person Management** - View, add, and remove enrolled persons
- **➕ Easy Enrollment** - Upload images through web interface
- **📊 System Statistics** - Database and performance metrics

### **API Endpoints:**
- **GET /** - Web interface
- **GET /health** - Health check
- **GET /persons** - List all persons
- **POST /enroll/images** - Enroll person from images
- **DELETE /persons/{name}** - Remove person
- **POST /identify/image** - Identify persons in image
- **GET /stream** - Live video stream
- **GET /docs** - API documentation (Swagger UI)

### **Test the API:**
```bash
python3 test_api.py
```

## 📋 **Command Line Usage**

### **Enrollment System**

#### **Enroll from Webcam:**
```bash
python3 simple_enrollment_organized.py enroll --name "John Doe" --webcam --captures 5
```

#### **Enroll from Images:**
```bash
python3 simple_enrollment_organized.py enroll --name "Jane Smith" --images photo1.jpg photo2.jpg photo3.jpg
```

#### **Enroll from Folder:**
```bash
python3 simple_enrollment_organized.py enroll --name "Mike Johnson" --folder /path/to/images/
```

#### **List Enrolled Persons:**
```bash
python3 simple_enrollment_organized.py list
```

#### **Remove Person:**
```bash
python3 simple_enrollment_organized.py remove --name "Person Name"
```

#### **Database Management:**
```bash
# Show statistics
python3 simple_enrollment_organized.py stats

# Create backup
python3 simple_enrollment_organized.py backup --backup-name "my_backup"
```

### **Identification System**

#### **Real-time Webcam Identification:**
```bash
python3 simple_identification_organized.py
# Press 'q' to quit, 's' to save frame, 'b' to backup database
```

#### **Process Video File:**
```bash
python3 simple_identification_organized.py --mode video --input input.mp4 --output output.mp4
```

#### **System Information:**
```bash
python3 simple_identification_organized.py --info
```

## ⚙️ **Configuration**

The system configuration is managed in `config.py`:

### **Key Settings:**
```python
# Detection settings
PERSON_DETECTION_CONFIDENCE = 0.4      # Person detection threshold
FACE_RECOGNITION_THRESHOLD = 0.6       # Face recognition threshold

# Database settings
DATABASE_PATH = "face_database/organized"  # Organized database path

# Enrollment settings
MIN_ENROLLMENT_IMAGES = 3               # Minimum images for enrollment
MAX_ENROLLMENT_IMAGES = 5               # Maximum images per person

# Visualization settings
BBOX_COLOR_KNOWN = (0, 255, 0)         # Green for known persons
BBOX_COLOR_UNKNOWN = (0, 0, 255)       # Red for unknown persons
```

## 🧪 **Testing**

### **Run System Tests:**
```bash
python3 test_organized_database.py
```

This will test:
- ✅ Database creation and structure
- ✅ Person enrollment and retrieval
- ✅ Face identification accuracy
- ✅ Backup functionality
- ✅ Migration capabilities

## 📊 **System Features**

### **Person Detection (YOLOv8)**
- Real-time person detection in video streams
- Configurable confidence thresholds
- Bounding box visualization
- Multiple person support

### **Face Processing (OpenCV)**
- Haar Cascade face detection
- ORB feature extraction
- Local Binary Pattern (LBP) histograms
- Cosine similarity matching

### **Organized Database**
- Individual person directories
- Unique person IDs (name_001, name_002, etc.)
- NumPy array storage for embeddings
- JSON metadata with rich information
- Reference image storage
- Master index for fast lookup

### **Advanced Features**
- **Live Backup**: Press 'b' during identification
- **Frame Saving**: Press 's' to save current frame
- **Database Statistics**: Comprehensive reporting
- **Migration Support**: Upgrade from legacy systems
- **Concurrent Access**: Multiple operations support

## 🔧 **File Structure**

```
Person Detection and Identification Model/
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── config.py                          # System configuration
├── yolov8n.pt                         # YOLOv8 model weights
│
├── Core System Files:
├── person_detector.py                 # YOLOv8 person detection
├── face_processor_simple.py           # OpenCV face processing
├── face_database_organized.py         # Organized database engine
│
├── Application Files:
├── simple_enrollment_organized.py     # Enrollment system
├── simple_identification_organized.py # Identification system
├── api_server.py                      # FastAPI web server
│
├── Utilities:
├── create_demo_faces.py               # Demo face generator
├── test_organized_database.py         # System tests
├── test_api.py                        # API server tests
│
└── Data:
    ├── face_database/                 # Database directory
    │   └── organized/                 # Organized database
    └── demo_faces/                    # Demo face images
```

## 🎨 **Demo Faces**

The system includes a demo face generator for testing:

```bash
python3 create_demo_faces.py
```

This creates synthetic face images for 4 demo persons (Alice, Bob, Charlie, Diana) that you can use to test the system without real photos.

## 📈 **Performance**

### **System Specifications:**
- **Person Detection**: YOLOv8n model (~6MB)
- **Face Processing**: OpenCV Haar Cascades + ORB features
- **Database**: Individual NumPy files per person
- **Memory Usage**: ~50-100MB during operation
- **FPS**: 15-30 FPS on standard hardware

### **Scalability:**
- ✅ **Persons**: Tested with 100+ persons
- ✅ **Embeddings**: 5-10 embeddings per person
- ✅ **Database Size**: Scales linearly
- ✅ **Lookup Speed**: Constant time O(1) per person

## 🛡️ **Data Safety**

### **Backup System:**
- **Automatic backups** during critical operations
- **Manual backup** command available
- **ZIP compression** for efficient storage
- **Timestamp-based** backup naming

### **Data Protection:**
- **Migration safety** with full backup before changes
- **Atomic operations** to prevent corruption
- **Index validation** on database operations
- **Error recovery** with rollback capabilities

## 🚨 **Troubleshooting**

### **Common Issues:**

#### **"No module named 'cv2'"**
```bash
pip3 install opencv-python
```

#### **"Could not open camera"**
- Check camera permissions
- Try different camera ID: `--camera 1`
- Ensure camera is not used by other applications

#### **"No faces detected"**
- Ensure good lighting
- Position face clearly in frame
- Adjust `FACE_DETECTION_CONFIDENCE` in config

#### **"YOLO model not found"**
- Ensure `yolov8n.pt` is in project directory
- Download manually: `wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt`

### **Performance Issues:**

#### **Low FPS:**
- Reduce input resolution
- Lower detection confidence thresholds
- Use GPU acceleration (install `torch` with CUDA)

#### **High Memory Usage:**
- Reduce `MAX_ENROLLMENT_IMAGES`
- Clear old backups periodically
- Use video file processing instead of real-time

## 🔮 **Future Enhancements**

### **Planned Features:**
- **GPU Acceleration** - CUDA support for faster processing
- **Deep Learning Models** - Integration with FaceNet/ArcFace
- **Object Tracking** - Deep SORT integration
- **Web Interface** - Browser-based management
- **REST API** - HTTP endpoints for integration
- **Multi-camera Support** - Multiple camera streams
- **Cloud Storage** - Database synchronization
- **Mobile App** - Android/iOS companion

### **Technical Improvements:**
- **Batch Processing** - Multiple face processing
- **Model Optimization** - TensorRT/ONNX conversion
- **Database Sharding** - Distributed storage
- **Caching System** - Redis integration
- **Monitoring** - Prometheus metrics
- **Security** - Encryption and authentication

## 📝 **Requirements**

### **Python Dependencies:**
```
ultralytics>=8.0.0          # YOLOv8 for person detection
opencv-python>=4.8.0        # Computer vision operations
numpy>=1.24.0               # Numerical computing
pillow>=10.0.0              # Image processing
loguru>=0.7.0               # Enhanced logging
scikit-learn>=1.3.0         # Machine learning utilities
torch>=2.0.0                # PyTorch for YOLO
torchvision>=0.15.0         # Computer vision for PyTorch
```

### **System Requirements:**
- **OS**: Linux, Windows, macOS
- **Python**: 3.8+
- **RAM**: 2GB+ recommended
- **Storage**: 100MB+ for models and database
- **Camera**: USB webcam (optional)

## 📄 **License**

This project is developed for educational and research purposes. Please ensure compliance with local privacy laws when using facial recognition technology.

## 🤝 **Contributing**

Contributions are welcome! Areas for improvement:
- Model accuracy enhancements
- Performance optimizations
- UI/UX improvements
- Documentation updates
- Bug fixes and testing

## 📞 **Support**

For issues and questions:
1. Check the troubleshooting section
2. Review system tests: `python3 test_organized_database.py`
3. Verify configuration in `config.py`
4. Check system info: `python3 simple_identification_organized.py --info`

---

**🎉 Ready to identify persons in real-time with an organized, scalable database system!**

**Quick Start Commands:**

**Desktop Application:**
```bash
python3 simple_identification_organized.py --info && python3 simple_identification_organized.py
```

**Web Interface:**
```bash
python3 api_server.py
# Open http://localhost:8000 in your browser

<img width="1242" height="762" alt="Screenshot 2025-08-16 015046" src="https://github.com/user-attachments/assets/9ef87449-7991-4f1f-aba2-1ec21a3d6de9" />
<img width="864" height="837" alt="Screenshot 2025-08-16 024831" src="https://github.com/user-attachments/assets/f72c98bb-06ae-4d32-9f8e-97f5d3517783" />
<img width="887" height="503" alt="Screenshot 2025-08-16 024840" src="https://github.com/user-attachments/assets/a008ddae-5eb7-4f60-996f-c3684956bcca" />

``` 
