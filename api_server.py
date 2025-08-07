#!/usr/bin/env python3
"""
Person Identification API Server
FastAPI-based REST API for person detection and identification system
"""

import os
import io
import cv2
import json
import base64
import asyncio
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

# FastAPI imports
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# System imports
from config import Config
from person_detector import PersonDetector
from face_processor_simple import SimpleFaceProcessor
from face_database_organized import OrganizedFaceDatabase
from simple_enrollment_organized import SimpleEnrollmentSystemOrganized
from simple_identification_organized import SimplePersonIdentificationSystemOrganized

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Person Identification API",
    description="Real-time person detection and identification system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global system instances
identification_system = None
enrollment_system = None
database = None
current_camera = None
streaming_active = False

# Pydantic models for API requests/responses
class PersonInfo(BaseModel):
    name: str
    person_id: str
    num_embeddings: int
    date_added: str
    metadata: Dict[str, Any]

class EnrollmentRequest(BaseModel):
    name: str
    additional_info: Optional[Dict[str, Any]] = None

class SystemInfo(BaseModel):
    system_name: str
    version: str
    database_stats: Dict[str, Any]
    config: Dict[str, Any]

class DetectionResult(BaseModel):
    person_id: int
    bbox: List[int]
    confidence: float
    identity: str
    person_db_id: Optional[str]
    face_similarity: float
    num_faces: int

class StreamFrame(BaseModel):
    frame: str  # base64 encoded
    detections: List[DetectionResult]
    timestamp: str

# Initialize system components
def initialize_system():
    """Initialize all system components"""
    global identification_system, enrollment_system, database
    
    try:
        logger.info("Initializing person identification system...")
        identification_system = SimplePersonIdentificationSystemOrganized()
        
        logger.info("Initializing enrollment system...")
        enrollment_system = SimpleEnrollmentSystemOrganized()
        
        logger.info("Initializing database...")
        database = OrganizedFaceDatabase()
        
        logger.info("System initialization complete!")
        
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        raise

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize system on startup"""
    initialize_system()

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global current_camera, streaming_active
    streaming_active = False
    if current_camera:
        current_camera.release()

# API Endpoints

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve basic web interface"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Person Identification System</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 1200px; margin: 0 auto; }
            .section { margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }
            .video-container { text-align: center; margin: 20px 0; }
            #videoStream { max-width: 100%; border: 2px solid #333; }
            button { padding: 10px 20px; margin: 5px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; }
            button:hover { background: #0056b3; }
            .person-list { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 15px; }
            .person-card { padding: 15px; border: 1px solid #ccc; border-radius: 8px; background: #f9f9f9; }
            .status { padding: 10px; border-radius: 4px; margin: 10px 0; }
            .success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
            .error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎯 Person Identification System</h1>
            
            <div class="section">
                <h2>📹 Live Video Stream</h2>
                <div class="video-container">
                    <img id="videoStream" src="/stream" alt="Video Stream" style="display:none;">
                    <div id="streamStatus">Click "Start Stream" to begin video feed</div>
                </div>
                <button onclick="startStream()">Start Stream</button>
                <button onclick="stopStream()">Stop Stream</button>
            </div>
            
            <div class="section">
                <h2>👥 Enrolled Persons</h2>
                <div id="personList" class="person-list"></div>
                <button onclick="loadPersons()">Refresh List</button>
            </div>
            
            <div class="section">
                <h2>➕ Add New Person</h2>
                <form id="enrollForm" enctype="multipart/form-data">
                    <input type="text" id="personName" placeholder="Person Name" required style="padding: 8px; margin: 5px; width: 200px;">
                    <input type="file" id="personImages" multiple accept="image/*" style="margin: 5px;">
                    <button type="submit">Enroll Person</button>
                </form>
                <div id="enrollStatus"></div>
            </div>
            
            <div class="section">
                <h2>📊 System Information</h2>
                <div id="systemInfo">Loading...</div>
                <button onclick="loadSystemInfo()">Refresh Info</button>
            </div>
        </div>

        <script>
            let streamActive = false;

            async function startStream() {
                try {
                    const response = await fetch('/start_stream', { method: 'POST' });
                    const result = await response.json();
                    
                    if (result.status === 'success') {
                        document.getElementById('videoStream').style.display = 'block';
                        document.getElementById('streamStatus').textContent = 'Stream active';
                        streamActive = true;
                    } else {
                        document.getElementById('streamStatus').textContent = 'Failed to start stream: ' + result.message;
                    }
                } catch (error) {
                    document.getElementById('streamStatus').textContent = 'Error starting stream: ' + error.message;
                }
            }

            async function stopStream() {
                try {
                    const response = await fetch('/stop_stream', { method: 'POST' });
                    const result = await response.json();
                    
                    document.getElementById('videoStream').style.display = 'none';
                    document.getElementById('streamStatus').textContent = 'Stream stopped';
                    streamActive = false;
                } catch (error) {
                    document.getElementById('streamStatus').textContent = 'Error stopping stream: ' + error.message;
                }
            }

            async function loadPersons() {
                try {
                    const response = await fetch('/persons');
                    const persons = await response.json();
                    
                    const container = document.getElementById('personList');
                    container.innerHTML = '';
                    
                    persons.forEach(person => {
                        const card = document.createElement('div');
                        card.className = 'person-card';
                        card.innerHTML = `
                            <h3>${person.name}</h3>
                            <p><strong>ID:</strong> ${person.person_id}</p>
                            <p><strong>Embeddings:</strong> ${person.num_embeddings}</p>
                            <p><strong>Added:</strong> ${new Date(person.date_added).toLocaleDateString()}</p>
                            <button onclick="removePerson('${person.name}')" style="background: #dc3545;">Remove</button>
                        `;
                        container.appendChild(card);
                    });
                } catch (error) {
                    document.getElementById('personList').innerHTML = 'Error loading persons: ' + error.message;
                }
            }

            async function removePerson(name) {
                if (!confirm(`Are you sure you want to remove ${name}?`)) return;
                
                try {
                    const response = await fetch(`/persons/${encodeURIComponent(name)}`, { method: 'DELETE' });
                    const result = await response.json();
                    
                    if (result.status === 'success') {
                        loadPersons(); // Refresh list
                    } else {
                        alert('Failed to remove person: ' + result.message);
                    }
                } catch (error) {
                    alert('Error removing person: ' + error.message);
                }
            }

            async function loadSystemInfo() {
                try {
                    const response = await fetch('/system/info');
                    const info = await response.json();
                    
                    document.getElementById('systemInfo').innerHTML = `
                        <p><strong>System:</strong> ${info.system_name} v${info.version}</p>
                        <p><strong>Database:</strong> ${info.database_stats.total_persons} persons, ${info.database_stats.total_embeddings} embeddings</p>
                        <p><strong>Database Size:</strong> ${info.database_stats.total_size_mb} MB</p>
                        <p><strong>Face Recognition Threshold:</strong> ${info.config.face_recognition_threshold}</p>
                    `;
                } catch (error) {
                    document.getElementById('systemInfo').innerHTML = 'Error loading system info: ' + error.message;
                }
            }

            document.getElementById('enrollForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const name = document.getElementById('personName').value;
                const files = document.getElementById('personImages').files;
                
                if (!name || files.length === 0) {
                    document.getElementById('enrollStatus').innerHTML = '<div class="error">Please provide name and at least one image</div>';
                    return;
                }
                
                const formData = new FormData();
                formData.append('name', name);
                
                for (let file of files) {
                    formData.append('images', file);
                }
                
                try {
                    document.getElementById('enrollStatus').innerHTML = '<div class="status">Enrolling person...</div>';
                    
                    const response = await fetch('/enroll/images', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (result.status === 'success') {
                        document.getElementById('enrollStatus').innerHTML = '<div class="success">Person enrolled successfully!</div>';
                        document.getElementById('enrollForm').reset();
                        loadPersons(); // Refresh person list
                    } else {
                        document.getElementById('enrollStatus').innerHTML = '<div class="error">Enrollment failed: ' + result.message + '</div>';
                    }
                } catch (error) {
                    document.getElementById('enrollStatus').innerHTML = '<div class="error">Error: ' + error.message + '</div>';
                }
            });

            // Load initial data
            loadPersons();
            loadSystemInfo();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/system/info", response_model=SystemInfo)
async def get_system_info():
    """Get system information and statistics"""
    try:
        info = identification_system.get_system_info()
        return SystemInfo(**info)
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/persons", response_model=List[PersonInfo])
async def get_persons():
    """Get list of all enrolled persons"""
    try:
        persons = database.get_all_persons()
        return [PersonInfo(**person) for person in persons]
    except Exception as e:
        logger.error(f"Error getting persons: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/persons/{person_name}")
async def get_person(person_name: str):
    """Get information about a specific person"""
    try:
        persons = database.get_all_persons()
        person = next((p for p in persons if p["name"].lower() == person_name.lower()), None)
        
        if not person:
            raise HTTPException(status_code=404, detail="Person not found")
        
        return PersonInfo(**person)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting person {person_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/persons/{person_name}")
async def remove_person(person_name: str):
    """Remove a person from the database"""
    try:
        success = enrollment_system.remove_person(person_name)
        
        if success:
            return {"status": "success", "message": f"Person '{person_name}' removed successfully"}
        else:
            raise HTTPException(status_code=404, detail="Person not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing person {person_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/enroll/images")
async def enroll_from_images(
    name: str = Form(...),
    images: List[UploadFile] = File(...),
    additional_info: str = Form(None)
):
    """Enroll a person using uploaded images"""
    try:
        if len(images) < Config.MIN_ENROLLMENT_IMAGES:
            raise HTTPException(
                status_code=400, 
                detail=f"At least {Config.MIN_ENROLLMENT_IMAGES} images required"
            )
        
        # Save uploaded images temporarily
        temp_dir = Path("temp_enrollment")
        temp_dir.mkdir(exist_ok=True)
        
        image_paths = []
        for i, image in enumerate(images):
            if not image.content_type.startswith('image/'):
                raise HTTPException(status_code=400, detail=f"File {image.filename} is not an image")
            
            # Save image
            image_path = temp_dir / f"{name}_{i}_{image.filename}"
            with open(image_path, "wb") as f:
                content = await image.read()
                f.write(content)
            image_paths.append(str(image_path))
        
        # Parse additional info
        extra_info = {}
        if additional_info:
            try:
                extra_info = json.loads(additional_info)
            except json.JSONDecodeError:
                logger.warning("Invalid additional_info JSON, ignoring")
        
        extra_info.update({
            "enrollment_method": "api_upload",
            "api_enrollment_time": datetime.now().isoformat(),
            "num_uploaded_images": len(images)
        })
        
        # Enroll person
        success = enrollment_system.enroll_from_images(name, image_paths, extra_info)
        
        # Cleanup temporary files
        for path in image_paths:
            try:
                os.remove(path)
            except:
                pass
        
        if success:
            return {"status": "success", "message": f"Person '{name}' enrolled successfully"}
        else:
            raise HTTPException(status_code=500, detail="Enrollment failed")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error enrolling person {name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/enroll/webcam")
async def enroll_from_webcam(request: EnrollmentRequest):
    """Start webcam enrollment (returns instructions for now)"""
    return {
        "status": "info",
        "message": "Webcam enrollment not implemented in API mode. Use the desktop application or upload images."
    }

@app.get("/database/stats")
async def get_database_stats():
    """Get database statistics"""
    try:
        stats = database.get_database_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/database/backup")
async def create_backup(backup_name: str = None):
    """Create database backup"""
    try:
        success = database.create_backup(backup_name)
        
        if success:
            return {"status": "success", "message": "Backup created successfully"}
        else:
            raise HTTPException(status_code=500, detail="Backup creation failed")
    except Exception as e:
        logger.error(f"Error creating backup: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/start_stream")
async def start_video_stream():
    """Start video streaming"""
    global current_camera, streaming_active
    
    try:
        if streaming_active:
            return {"status": "info", "message": "Stream already active"}
        
        # Initialize camera
        current_camera = cv2.VideoCapture(0)
        if not current_camera.isOpened():
            raise HTTPException(status_code=500, detail="Could not open camera")
        
        # Set camera properties
        current_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        current_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        current_camera.set(cv2.CAP_PROP_FPS, 30)
        
        streaming_active = True
        return {"status": "success", "message": "Video stream started"}
    
    except Exception as e:
        logger.error(f"Error starting video stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stop_stream")
async def stop_video_stream():
    """Stop video streaming"""
    global current_camera, streaming_active
    
    try:
        streaming_active = False
        if current_camera:
            current_camera.release()
            current_camera = None
        
        return {"status": "success", "message": "Video stream stopped"}
    
    except Exception as e:
        logger.error(f"Error stopping video stream: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def generate_frames():
    """Generate video frames for streaming"""
    global current_camera, streaming_active
    
    while streaming_active and current_camera:
        try:
            ret, frame = current_camera.read()
            if not ret:
                break
            
            # Process frame for person identification
            processed_frame, detections = identification_system.process_frame(frame)
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            if not ret:
                continue
            
            frame_bytes = buffer.tobytes()
            
            # Yield frame in multipart format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        except Exception as e:
            logger.error(f"Error in frame generation: {e}")
            break

@app.get("/stream")
async def video_stream():
    """Video streaming endpoint"""
    global streaming_active
    
    if not streaming_active:
        raise HTTPException(status_code=400, detail="Video stream not active. Start stream first.")
    
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/frame/current")
async def get_current_frame():
    """Get current frame with detection results as JSON"""
    global current_camera, streaming_active
    
    try:
        if not streaming_active or not current_camera:
            raise HTTPException(status_code=400, detail="Video stream not active")
        
        ret, frame = current_camera.read()
        if not ret:
            raise HTTPException(status_code=500, detail="Could not read frame")
        
        # Process frame
        processed_frame, detections = identification_system.process_frame(frame)
        
        # Encode frame as base64
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        if not ret:
            raise HTTPException(status_code=500, detail="Could not encode frame")
        
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Convert detections to API format
        api_detections = []
        for detection in detections:
            api_detections.append(DetectionResult(
                person_id=detection['person_id'],
                bbox=list(detection['bbox']),
                confidence=detection['confidence'],
                identity=detection['identity'],
                person_db_id=detection['person_db_id'],
                face_similarity=detection['face_similarity'],
                num_faces=detection['num_faces']
            ))
        
        return StreamFrame(
            frame=frame_base64,
            detections=api_detections,
            timestamp=datetime.now().isoformat()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting current frame: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/identify/image")
async def identify_from_image(image: UploadFile = File(...)):
    """Identify persons in an uploaded image"""
    try:
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File is not an image")
        
        # Read and decode image
        image_bytes = await image.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        
        # Process frame
        processed_frame, detections = identification_system.process_frame(frame)
        
        # Encode processed frame
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        if not ret:
            raise HTTPException(status_code=500, detail="Could not encode processed image")
        
        processed_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Convert detections to API format
        api_detections = []
        for detection in detections:
            api_detections.append(DetectionResult(
                person_id=detection['person_id'],
                bbox=list(detection['bbox']),
                confidence=detection['confidence'],
                identity=detection['identity'],
                person_db_id=detection['person_db_id'],
                face_similarity=detection['face_similarity'],
                num_faces=detection['num_faces']
            ))
        
        return {
            "status": "success",
            "processed_image": processed_base64,
            "detections": api_detections,
            "num_persons_detected": len(detections),
            "identified_persons": [d.identity for d in api_detections if d.identity != "Unknown"]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error identifying from image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check system components
        system_ok = identification_system is not None
        database_ok = database is not None
        enrollment_ok = enrollment_system is not None
        
        # Get basic stats
        stats = database.get_database_stats() if database else {}
        
        return {
            "status": "healthy" if all([system_ok, database_ok, enrollment_ok]) else "unhealthy",
            "components": {
                "identification_system": system_ok,
                "database": database_ok,
                "enrollment_system": enrollment_ok
            },
            "database_stats": {
                "total_persons": stats.get("total_persons", 0),
                "total_embeddings": stats.get("total_embeddings", 0)
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Main function to run the server
def main():
    """Run the API server"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Person Identification API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    logger.info(f"Starting Person Identification API Server on {args.host}:{args.port}")
    logger.info(f"Web interface will be available at http://{args.host}:{args.port}")
    
    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )

if __name__ == "__main__":
    main() 