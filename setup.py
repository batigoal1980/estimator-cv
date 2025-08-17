#!/usr/bin/env python3
"""
Setup script for Estimator CV Demo
Helps with installation and environment setup
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    if sys.version_info < (3, 8):
        print(f"❌ Python {sys.version} is not supported. Please use Python 3.8 or higher.")
        return False
    else:
        print(f"✅ Python {sys.version} is compatible")
        return True

def install_python_dependencies():
    """Install Python dependencies"""
    print("📦 Installing Python dependencies...")
    
    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found")
        return False
    
    # Install dependencies
    if not run_command("pip install -r requirements.txt", "Installing Python packages"):
        return False
    
    return True

def install_system_dependencies():
    """Install system dependencies based on platform"""
    system = platform.system().lower()
    
    print(f"💻 Installing system dependencies for {system}...")
    
    if system == "darwin":  # macOS
        # Check if Homebrew is installed
        if not run_command("which brew", "Checking Homebrew installation"):
            print("📥 Installing Homebrew...")
            install_brew_cmd = '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
            if not run_command(install_brew_cmd, "Installing Homebrew"):
                return False
        
        # Install required packages
        packages = ["tesseract", "poppler", "opencv"]
        for package in packages:
            if not run_command(f"brew install {package}", f"Installing {package}"):
                print(f"⚠️  Failed to install {package}, continuing...")
        
    elif system == "linux":
        # Check if apt is available (Ubuntu/Debian)
        if run_command("which apt", "Checking apt package manager"):
            packages = ["tesseract-ocr", "poppler-utils", "libopencv-dev", "python3-opencv"]
            for package in packages:
                if not run_command(f"sudo apt-get install -y {package}", f"Installing {package}"):
                    print(f"⚠️  Failed to install {package}, continuing...")
        
        # Check if yum is available (CentOS/RHEL)
        elif run_command("which yum", "Checking yum package manager"):
            packages = ["tesseract", "poppler-utils", "opencv-devel"]
            for package in packages:
                if not run_command(f"sudo yum install -y {package}", f"Installing {package}"):
                    print(f"⚠️  Failed to install {package}, continuing...")
        
        else:
            print("⚠️  Unsupported Linux distribution. Please install tesseract, poppler, and opencv manually.")
    
    elif system == "windows":
        print("⚠️  Windows detected. Please install the following manually:")
        print("   • Tesseract OCR: https://github.com/UB-Mannheim/tesseract/wiki")
        print("   • Poppler: http://blog.alivate.com.au/poppler-windows/")
        print("   • OpenCV: pip install opencv-python")
    
    return True

def download_model_checkpoints():
    """Download model checkpoints if needed"""
    print("🤖 Setting up model checkpoints...")
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    print("📝 Note: Model checkpoints need to be downloaded manually:")
    print("   • SAM2: Download from Meta AI")
    print("   • Grounding-DINO: Download from IDEA Research")
    print("   • Place checkpoints in the 'models' directory")
    
    return True

def create_sample_data():
    """Create sample data for testing"""
    print("📊 Creating sample data...")
    
    try:
        from utils import create_demo_sample_pdf
        sample_pdf = create_demo_sample_pdf()
        if sample_pdf:
            print(f"✅ Created sample PDF: {sample_pdf}")
        else:
            print("⚠️  Could not create sample PDF")
    except ImportError:
        print("⚠️  Utils module not available")
    
    return True

def run_system_check():
    """Run a system check to verify installation"""
    print("🔍 Running system check...")
    
    try:
        from utils import check_system_requirements, print_system_info
        print_system_info()
        return True
    except ImportError:
        print("❌ Could not run system check")
        return False

def main():
    """Main setup function"""
    print("🚀 ESTIMATOR CV DEMO - SETUP")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install Python dependencies
    if not install_python_dependencies():
        print("❌ Failed to install Python dependencies")
        sys.exit(1)
    
    # Install system dependencies
    if not install_system_dependencies():
        print("⚠️  Some system dependencies failed to install")
    
    # Download model checkpoints
    download_model_checkpoints()
    
    # Create sample data
    create_sample_data()
    
    # Run system check
    run_system_check()
    
    print("\n🎉 Setup completed!")
    print("\n📚 Next steps:")
    print("1. Place a PDF file in this directory")
    print("2. Run: python quick_start.py")
    print("3. Or run individual components:")
    print("   • python pdf_processor.py --input file.pdf --output images/")
    print("   • python layout_detector.py --input images/ --output layouts/")
    print("   • python sam2_grounding.py --input images/ --output segments/")
    print("   • python cubicasa_demo.py --input images/ --output classified/")
    
    print("\n🔧 For real-time processing:")
    print("   • Download model checkpoints to 'models' directory")
    print("   • Set ROBOFLOW_API_KEY environment variable")
    print("   • Run with --real-api flag")

if __name__ == "__main__":
    main()
