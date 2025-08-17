#!/usr/bin/env python3
"""
Simple script to start the Estimator CV Demo web interface
"""

import os
import sys
import subprocess
import webbrowser
import time

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import flask
        print("✅ Flask is installed")
        return True
    except ImportError:
        print("❌ Flask is not installed")
        print("Installing Flask...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "Flask==2.3.3"])
            print("✅ Flask installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("❌ Failed to install Flask")
            return False

def create_directories():
    """Create necessary directories"""
    dirs = ['uploads', 'web_results', 'templates']
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        print(f"✅ Created directory: {dir_name}")

def start_web_interface():
    """Start the Flask web interface"""
    print("🚀 Starting Estimator CV Demo Web Interface...")
    print("📁 Upload folder: uploads/")
    print("📁 Results folder: web_results/")
    print("🌐 Web interface will be available at: http://localhost:5000")
    
    # Open browser after a short delay
    def open_browser():
        time.sleep(2)
        try:
            webbrowser.open('http://localhost:5000')
            print("🌐 Opened web browser automatically")
        except:
            print("🌐 Please open your browser and go to: http://localhost:5000")
    
    import threading
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Start Flask app
    try:
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except ImportError as e:
        print(f"❌ Error importing app: {e}")
        print("Make sure all required files are in place")
        return False
    except Exception as e:
        print(f"❌ Error starting web interface: {e}")
        return False

def main():
    """Main function"""
    print("=" * 60)
    print("🌐 ESTIMATOR CV DEMO - WEB INTERFACE")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Cannot start web interface due to missing dependencies")
        return
    
    # Create directories
    create_directories()
    
    # Start web interface
    start_web_interface()

if __name__ == "__main__":
    main()
