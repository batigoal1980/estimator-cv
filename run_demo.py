#!/usr/bin/env python3
"""
Auto-Run Script for Estimator CV Demo
This script will automatically set up and run everything!
"""

import os
import sys
import subprocess
import webbrowser
import time

def run_command(command, description):
    """Run a command and show progress"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        return False

def install_dependencies():
    """Install all required dependencies"""
    print("📦 Installing dependencies...")
    
    dependencies = [
        "Flask==2.3.3",
        "opencv-python",
        "numpy",
        "matplotlib",
        "Pillow",
        "requests"
    ]
    
    for dep in dependencies:
        print(f"  Installing {dep}...")
        if not run_command(f"pip install {dep}", f"Installing {dep}"):
            print(f"⚠️  Warning: Failed to install {dep}")
    
    return True

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    dirs = ['uploads', 'web_results', 'templates']
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        print(f"✅ Created: {dir_name}/")

def start_web_interface():
    """Start the web interface"""
    print("🚀 Starting web interface...")
    
    # Check if app.py exists
    if not os.path.exists('app.py'):
        print("❌ Error: app.py not found!")
        print("Make sure you're in the Estimator-CV directory")
        return False
    
    # Open browser after delay
    def open_browser():
        time.sleep(3)
        try:
            webbrowser.open('http://localhost:5000')
            print("🌐 Opened browser automatically!")
        except:
            print("🌐 Please open your browser and go to: http://localhost:5000")
    
    import threading
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Start Flask app
    try:
        print("🌐 Starting Flask server...")
        print("📱 Web interface will be available at: http://localhost:5000")
        print("⏹️  Press Ctrl+C to stop the server")
        print("=" * 60)
        
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
        
    except ImportError as e:
        print(f"❌ Error importing app: {e}")
        print("Trying to install missing dependencies...")
        install_dependencies()
        print("Please run this script again after installation completes")
        return False
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
        return True
    except Exception as e:
        print(f"❌ Error starting web interface: {e}")
        return False

def main():
    """Main function - runs everything automatically!"""
    print("=" * 60)
    print("🚀 ESTIMATOR CV DEMO - AUTO SETUP & RUN")
    print("=" * 60)
    print("This script will automatically:")
    print("1. Install all dependencies")
    print("2. Create necessary directories")
    print("3. Start the web interface")
    print("4. Open your browser")
    print("=" * 60)
    
    # Install dependencies
    install_dependencies()
    
    # Create directories
    create_directories()
    
    # Start web interface
    start_web_interface()

if __name__ == "__main__":
    main()
