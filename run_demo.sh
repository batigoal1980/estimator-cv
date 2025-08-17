#!/bin/bash

echo "============================================================"
echo "🚀 ESTIMATOR CV DEMO - AUTO LAUNCHER"
echo "============================================================"
echo ""
echo "This will automatically:"
echo "1. Install dependencies"
echo "2. Start the web interface"
echo "3. Open your browser"
echo ""
echo "Press Enter to continue..."
read

echo ""
echo "📦 Installing dependencies..."
pip3 install Flask==2.3.3 opencv-python numpy matplotlib Pillow requests

echo ""
echo "📁 Creating directories..."
mkdir -p uploads web_results templates

echo ""
echo "🚀 Starting web interface..."
echo "🌐 Web interface will be available at: http://localhost:5000"
echo "⏹️  Press Ctrl+C to stop the server"
echo ""

# Open browser after delay
(sleep 3 && open http://localhost:5000) &

echo "🎉 Starting Flask server..."
python3 app.py
