#!/usr/bin/env python3
"""Test if coordinates are rotated"""

from roboflow import Roboflow
from PIL import Image
import cv2
import numpy as np

# Initialize Roboflow
rf = Roboflow(api_key="ZdkZuHA7GF5NdOfCwQo8")
model = rf.workspace("floorplan-recognition").project("cubicasa5k-2-qpmsa").version(3).model

# Test image
test_image = "web_results/session_20250815_233455/01_pdf_processed/page_001.png"

# Get image dimensions
img = Image.open(test_image)
width, height = img.size
print(f"📏 Image dimensions: {width} x {height}")
print(f"   Aspect ratio: {width/height:.2f}")

# Run inference
result = model.predict(test_image, confidence=25, overlap=50).json()

# Check API dimensions
api_width = result.get('image', {}).get('width')
api_height = result.get('image', {}).get('height')
print(f"📐 API dimensions: {api_width} x {api_height}")

# Check if dimensions are swapped
if api_width == height and api_height == width:
    print("⚠️  DIMENSIONS ARE SWAPPED! Image is rotated 90 degrees")
    
# Get first few predictions
print("\n🔍 First 3 detections (checking if rotated):")
for i, pred in enumerate(result.get('predictions', [])[:3]):
    x = pred['x']
    y = pred['y']
    w = pred['width']
    h = pred['height']
    
    print(f"\nDetection {i+1} ({pred['class']}):")
    print(f"  Center: ({x}, {y})")
    print(f"  Size: {w} x {h}")
    
    # Check if coordinates make sense
    if x > width or y > height:
        print(f"  ⚠️ Coordinates exceed image bounds!")
        
    # Try rotating coordinates
    rotated_x = y  # y becomes new x
    rotated_y = width - x  # width - x becomes new y
    print(f"  Rotated 90° CCW: ({rotated_x}, {rotated_y})")
    
    # Or rotate the other way
    rotated_x2 = height - y  # height - y becomes new x  
    rotated_y2 = x  # x becomes new y
    print(f"  Rotated 90° CW: ({rotated_x2}, {rotated_y2})")
