#!/usr/bin/env python3
"""Check what resolution Roboflow is using"""

from roboflow import Roboflow
from PIL import Image
from pathlib import Path

# Initialize Roboflow
rf = Roboflow(api_key="ZdkZuHA7GF5NdOfCwQo8")

# Access model
project = rf.workspace("floorplan-recognition").project("cubicasa5k-2-qpmsa")
model = project.version(3).model

# Find test image
test_image = "web_results/session_20250815_233455/01_pdf_processed/page_001.png"

# Get actual image dimensions
with Image.open(test_image) as img:
    actual_width, actual_height = img.size
    print(f"📏 Actual image dimensions: {actual_width} x {actual_height}")

# Run inference
result = model.predict(test_image, confidence=25, overlap=50).json()

# Check what dimensions the API thinks the image has
api_width = result.get('image', {}).get('width', 0)
api_height = result.get('image', {}).get('height', 0)
print(f"📐 API reported dimensions: {api_width} x {api_height}")

# Check if coordinates need scaling
if result.get('predictions'):
    pred = result['predictions'][0]
    print(f"\n🔍 First detection:")
    print(f"  x: {pred['x']}, y: {pred['y']}")
    print(f"  width: {pred['width']}, height: {pred['height']}")
    
    # Check if coordinates are out of bounds
    if pred['x'] > actual_width or pred['y'] > actual_height:
        print(f"\n⚠️  Coordinates exceed actual image dimensions!")
        print(f"  Scaling factor needed: {pred['x'] / actual_width:.2f}x")
