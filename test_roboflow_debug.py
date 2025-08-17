#!/usr/bin/env python3
"""Debug script to check Roboflow API response format"""

import json
from roboflow import Roboflow
from pathlib import Path

# Initialize Roboflow
rf = Roboflow(api_key="ZdkZuHA7GF5NdOfCwQo8")

# Access model
project = rf.workspace("floorplan-recognition").project("cubicasa5k-2-qpmsa")
model = project.version(3).model

# Find test image
test_images = list(Path("web_results").glob("*/01_pdf_processed/*.png"))
if test_images:
    image_path = str(test_images[0])
    print(f"Testing with: {image_path}")
    
    # Run inference with 25% confidence, 50% overlap (matching cloud settings)
    print("\n🔍 Running inference with confidence=25, overlap=50...")
    result = model.predict(image_path, confidence=25, overlap=50)
    
    # Get raw JSON to inspect format
    result_json = result.json()
    
    # Print first few predictions to see the format
    print("\n📊 Raw prediction format (first 3):")
    for i, pred in enumerate(result_json.get('predictions', [])[:3]):
        print(f"\nPrediction {i+1}:")
        print(f"  class: {pred.get('class')}")
        print(f"  confidence: {pred.get('confidence')} (type: {type(pred.get('confidence'))})")
        print(f"  x: {pred.get('x')}")
        print(f"  y: {pred.get('y')}")
        print(f"  width: {pred.get('width')}")
        print(f"  height: {pred.get('height')}")
    
    # Count total detections
    total = len(result_json.get('predictions', []))
    print(f"\n✅ Total detections: {total}")
    
    # Check confidence range
    confidences = [p['confidence'] for p in result_json.get('predictions', [])]
    if confidences:
        print(f"📈 Confidence range: {min(confidences):.2f} - {max(confidences):.2f}")
else:
    print("❌ No test images found")
