#!/usr/bin/env python3
"""
Utility functions for Estimator CV Demo
Helper functions and common operations
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
# import cv2  # Commented out for Railway compatibility
import numpy as np
from PIL import Image
import os

# Try to import cv2, but don't fail if it's not available
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    if os.getenv('RAILWAY_ENVIRONMENT'):
        print("CV2 not available on Railway - using PIL fallbacks")
    else:
        print("Warning: CV2 not available - some functions may not work")

logger = logging.getLogger(__name__)

def ensure_directory(path: str) -> str:
    """Ensure directory exists, create if it doesn't"""
    os.makedirs(path, exist_ok=True)
    return path

def get_image_info(image_path: str) -> Dict[str, Any]:
    """Get basic information about an image"""
    try:
        with Image.open(image_path) as img:
            info = {
                'path': image_path,
                'size': img.size,
                'mode': img.mode,
                'format': img.format,
                'width': img.width,
                'height': img.height,
                'file_size_mb': os.path.getsize(image_path) / (1024 * 1024)
            }
        return info
    except Exception as e:
        logger.error(f"Error getting image info for {image_path}: {e}")
        return {}

def resize_image(image_path: str, output_path: str, max_dimension: int = 1024) -> str:
    """Resize image while maintaining aspect ratio"""
    try:
        with Image.open(image_path) as img:
            # Calculate new dimensions
            width, height = img.size
            if width > height:
                new_width = max_dimension
                new_height = int(height * max_dimension / width)
            else:
                new_height = max_dimension
                new_width = int(width * max_dimension / height)
            
            # Resize image
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            resized_img.save(output_path, quality=95)
            
            logger.info(f"Resized {image_path} from {img.size} to {resized_img.size}")
            return output_path
            
    except Exception as e:
        logger.error(f"Error resizing image {image_path}: {e}")
        raise

def create_image_grid(image_paths: List[str], output_path: str, 
                     grid_size: tuple = (2, 2), max_dimension: int = 800) -> str:
    """Create a grid of images for comparison"""
    try:
        # Load and resize images
        images = []
        for img_path in image_paths[:grid_size[0] * grid_size[1]]:
            with Image.open(img_path) as img:
                # Resize to max_dimension
                img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
                images.append(img)
        
        # Calculate grid dimensions
        cell_width = max(img.width for img in images)
        cell_height = max(img.height for img in images)
        grid_width = cell_width * grid_size[1]
        grid_height = cell_height * grid_size[0]
        
        # Create grid image
        grid_img = Image.new('RGB', (grid_width, grid_height), 'white')
        
        # Place images in grid
        for i, img in enumerate(images):
            row = i // grid_size[1]
            col = i % grid_size[1]
            x = col * cell_width
            y = row * cell_height
            
            # Center image in cell
            paste_x = x + (cell_width - img.width) // 2
            paste_y = y + (cell_height - img.height) // 2
            
            grid_img.paste(img, (paste_x, paste_y))
        
        # Save grid
        grid_img.save(output_path, quality=95)
        logger.info(f"Created image grid: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error creating image grid: {e}")
        raise

def save_results_json(results: Dict[str, Any], output_path: str) -> str:
    """Save results to JSON file with proper formatting"""
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save with pretty formatting
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error saving results to JSON: {e}")
        raise

def load_results_json(json_path: str) -> Dict[str, Any]:
    """Load results from JSON file"""
    try:
        with open(json_path, 'r') as f:
            results = json.load(f)
        logger.info(f"Results loaded from: {json_path}")
        return results
        
    except Exception as e:
        logger.error(f"Error loading results from JSON: {e}")
        raise

def create_demo_sample_pdf() -> str:
    """Create a sample PDF for demo purposes if none exists"""
    try:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import letter
        
        sample_pdf = "demo_sample.pdf"
        
        # Create a simple PDF with some architectural elements
        c = canvas.Canvas(sample_pdf, pagesize=letter)
        width, height = letter
        
        # Title
        c.setFont("Helvetica-Bold", 24)
        c.drawString(50, height - 50, "Sample Floor Plan")
        
        # Simple floor plan elements
        c.setFont("Helvetica", 12)
        
        # Room 1
        c.rect(100, height - 200, 150, 100)
        c.drawString(110, height - 190, "Living Room")
        
        # Room 2
        c.rect(300, height - 200, 150, 100)
        c.drawString(310, height - 190, "Kitchen")
        
        # Door
        c.rect(250, height - 150, 20, 30)
        c.drawString(255, height - 140, "D")
        
        # Window
        c.rect(120, height - 120, 40, 20)
        c.drawString(125, height - 115, "W")
        
        # Dimensions
        c.drawString(100, height - 220, "15' x 10'")
        c.drawString(300, height - 220, "12' x 8'")
        
        c.save()
        
        logger.info(f"Created sample PDF: {sample_pdf}")
        return sample_pdf
        
    except ImportError:
        logger.warning("reportlab not available, cannot create sample PDF")
        return None
    except Exception as e:
        logger.error(f"Error creating sample PDF: {e}")
        return None

def validate_pdf_file(pdf_path: str) -> bool:
    """Validate that a file is a valid PDF"""
    try:
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file does not exist: {pdf_path}")
            return False
        
        if not pdf_path.lower().endswith('.pdf'):
            logger.error(f"File is not a PDF: {pdf_path}")
            return False
        
        # Try to open with PyPDF2 to validate
        try:
            import PyPDF2
            with open(pdf_path, 'rb') as f:
                PyPDF2.PdfReader(f)
            return True
        except ImportError:
            # PyPDF2 not available, just check file extension
            return True
        except Exception as e:
            logger.error(f"PDF validation failed: {e}")
            return False
            
    except Exception as e:
        logger.error(f"Error validating PDF: {e}")
        return False

def get_available_models() -> Dict[str, List[str]]:
    """Get list of available models for each component"""
    models = {
        'layout_detection': [
            'PubLayNet',
            'DocBank',
            'FUNSD'
        ],
        'segmentation': [
            'SAM2',
            'Grounding-DINO'
        ],
        'classification': [
            'CubiCasa5k',
            'Custom'
        ]
    }
    return models

def check_system_requirements() -> Dict[str, bool]:
    """Check if system meets requirements for the demo"""
    requirements = {
        'python_version': sys.version_info >= (3, 8),
        'opencv': False,
        'pillow': False,
        'numpy': False,
        'matplotlib': False
    }
    
    requirements['opencv'] = CV2_AVAILABLE
    
    try:
        import PIL
        requirements['pillow'] = True
    except ImportError:
        pass
    
    try:
        import numpy
        requirements['numpy'] = True
    except ImportError:
        pass
    
    try:
        import matplotlib
        requirements['matplotlib'] = True
    except ImportError:
        pass
    
    return requirements

def print_system_info():
    """Print system information and requirements status"""
    print("=" * 60)
    print("SYSTEM REQUIREMENTS CHECK")
    print("=" * 60)
    
    # Python version
    print(f"Python Version: {sys.version}")
    
    # System info
    import platform
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.architecture()}")
    
    # Requirements check
    requirements = check_system_requirements()
    print("\nRequired Packages:")
    for package, available in requirements.items():
        status = "✅" if available else "❌"
        print(f"  {status} {package}")
    
    # Recommendations
    print("\nRecommendations:")
    if not all(requirements.values()):
        print("  • Install missing packages: pip install -r requirements.txt")
    print("  • For GPU acceleration: Install CUDA-compatible PyTorch")
    print("  • For real-time processing: Get Roboflow API key")
    
    print("=" * 60)

if __name__ == "__main__":
    print_system_info()
