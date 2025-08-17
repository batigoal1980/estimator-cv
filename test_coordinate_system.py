#!/usr/bin/env python3
"""Debug coordinate system and bounding box placement"""

from roboflow import Roboflow
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

# Initialize Roboflow
rf = Roboflow(api_key="ZdkZuHA7GF5NdOfCwQo8")
model = rf.workspace("floorplan-recognition").project("cubicasa5k-2-qpmsa").version(3).model

# Test image
test_image = "web_results/session_20250815_233455/01_pdf_processed/page_001.png"

# Load image with PIL
pil_img = Image.open(test_image)
width, height = pil_img.size
print(f"📏 Image dimensions: {width} x {height}")

# Run inference with 30% confidence for moderate detection
result = model.predict(test_image, confidence=30, overlap=30).json()

# Get API dimensions
api_width = result.get('image', {}).get('width')
api_height = result.get('image', {}).get('height')
print(f"📐 API dimensions: {api_width} x {api_height}")

# Check first few predictions in detail
predictions = result.get('predictions', [])
print(f"\n🔍 Total detections: {len(predictions)}")

if predictions:
    # Create a copy for drawing
    draw_img = pil_img.copy()
    draw = ImageDraw.Draw(draw_img)
    
    # Draw first 5 predictions with different colors
    colors = ['red', 'blue', 'green', 'yellow', 'purple']
    
    for i, pred in enumerate(predictions[:5]):
        color = colors[i % len(colors)]
        
        x = pred['x']
        y = pred['y']
        w = pred['width']
        h = pred['height']
        conf = pred['confidence']
        cls = pred['class']
        
        print(f"\n{i+1}. {cls} (conf: {conf:.2f}):")
        print(f"   Center: ({x:.1f}, {y:.1f})")
        print(f"   Size: {w:.1f} x {h:.1f}")
        
        # Calculate corners (assuming x,y is center)
        x1 = x - w/2
        y1 = y - h/2
        x2 = x + w/2
        y2 = y + h/2
        
        print(f"   Top-left: ({x1:.1f}, {y1:.1f})")
        print(f"   Bottom-right: ({x2:.1f}, {y2:.1f})")
        
        # Check if within bounds
        if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
            print(f"   ⚠️ Box extends outside image bounds!")
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw center point
        draw.ellipse([x-5, y-5, x+5, y+5], fill=color)
        
        # Add label
        draw.text((x1, y1-20), f"{i+1}: {cls}", fill=color)
    
    # Save debug image
    draw_img.save('coordinate_debug.png')
    print("\n✅ Saved coordinate_debug.png with first 5 detections")
    
    # Also check if coordinates might be in a different format
    print("\n🔧 Checking coordinate format...")
    
    # Check if maybe x,y are top-left instead of center
    pred = predictions[0]
    x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
    
    print(f"If (x,y) is top-left: box would be ({x}, {y}) to ({x+w}, {y+h})")
    print(f"If (x,y) is center: box would be ({x-w/2}, {y-h/2}) to ({x+w/2}, {y+h/2})")
    
    # Create comparison image
    comparison = pil_img.copy()
    draw2 = ImageDraw.Draw(comparison)
    
    # Draw as center (red)
    draw2.rectangle([x-w/2, y-h/2, x+w/2, y+h/2], outline='red', width=3)
    draw2.text((x-w/2, y-h/2-20), "Center", fill='red')
    
    # Draw as top-left (blue)
    draw2.rectangle([x, y, x+w, y+h], outline='blue', width=3)
    draw2.text((x, y-20), "Top-left", fill='blue')
    
    comparison.save('coordinate_comparison.png')
    print("✅ Saved coordinate_comparison.png comparing interpretations")
