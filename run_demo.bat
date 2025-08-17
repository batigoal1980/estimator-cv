@echo off
echo ============================================================
echo 🚀 ESTIMATOR CV DEMO - AUTO LAUNCHER
echo ============================================================
echo.
echo This will automatically:
echo 1. Install dependencies
echo 2. Start the web interface
echo 3. Open your browser
echo.
echo Press any key to continue...
pause >nul

echo.
echo 📦 Installing dependencies...
pip install Flask==2.3.3 opencv-python numpy matplotlib Pillow requests

echo.
echo 📁 Creating directories...
if not exist "uploads" mkdir uploads
if not exist "web_results" mkdir web_results
if not exist "templates" mkdir templates

echo.
echo 🚀 Starting web interface...
echo 🌐 Web interface will be available at: http://localhost:5000
echo ⏹️  Press Ctrl+C to stop the server
echo.
echo Opening browser in 3 seconds...
timeout /t 3 /nobreak >nul
start http://localhost:5000

echo.
echo 🎉 Starting Flask server...
python app.py

pause
