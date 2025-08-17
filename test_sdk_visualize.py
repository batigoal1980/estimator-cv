#!/usr/bin/env python3
"""Use Roboflow SDK's built-in visualization"""

from roboflow import Roboflow
from pathlib import Path

# Initialize Roboflow
rf = Roboflow(api_key="ZdkZuHA7GF5NdOfCwQo8")
model = rf.workspace("floorplan-recognition").project("cubicasa5k-2-qpmsa").version(3).model

# Test image
test_image = "web_results/session_20250815_233455/01_pdf_processed/page_001.png"

print("🔍 Running CubiCasa5k detection with SDK visualization...")

# Run inference and save visualization directly
# The SDK should handle coordinates correctly
result = model.predict(test_image, confidence=30, overlap=30)

# Save the visualization using SDK's built-in method
result.save("sdk_visualization.jpg")

print("✅ Saved sdk_visualization.jpg using Roboflow SDK's built-in visualization")

# Also get the raw predictions to compare
predictions = result.json()['predictions']
print(f"📊 Found {len(predictions)} detections")

# Count by class
class_counts = {}
for pred in predictions:
    cls = pred['class']
    class_counts[cls] = class_counts.get(cls, 0) + 1

print("\n📈 Detection Summary:")
for cls, count in sorted(class_counts.items()):
    print(f"  • {cls}: {count}")
