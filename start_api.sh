#!/bin/bash
# Person Identification API Server Startup Script

echo "🚀 Starting Person Identification API Server..."
echo "=================================================="

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if required files exist
if [ ! -f "api_server.py" ]; then
    echo "❌ api_server.py not found in current directory"
    exit 1
fi

if [ ! -f "yolov8n.pt" ]; then
    echo "❌ yolov8n.pt model file not found"
    echo "   Please ensure the YOLO model is in the project directory"
    exit 1
fi

# Create demo faces if they don't exist
if [ ! -d "demo_faces" ]; then
    echo "📸 Creating demo faces for testing..."
    python3 create_demo_faces.py
fi

# Install dependencies if needed
echo "📦 Checking dependencies..."
python3 -c "import fastapi, uvicorn" 2>/dev/null || {
    echo "⚠️  FastAPI dependencies not found. Installing..."
    pip3 install fastapi uvicorn python-multipart aiofiles
}

# Check if port 8000 is already in use
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  Port 8000 is already in use"
    echo "   Kill existing process or use a different port:"
    echo "   python3 api_server.py --port 8001"
    exit 1
fi

echo ""
echo "✅ Starting API server..."
echo "   📹 Web Interface: http://localhost:8000"
echo "   📚 API Docs: http://localhost:8000/docs"
echo "   🏥 Health Check: http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server
python3 api_server.py --host 0.0.0.0 --port 8000 