#!/usr/bin/env python3
"""
Quick test script for CubiCasa5k-2 model
"""

import os
import sys
from cubicasa_demo import CubiCasaClassifier
from pathlib import Path

def test_cubicasa():
    # Check for API key
    api_key = os.getenv('ROBOFLOW_API_KEY')
    if not api_key:
        print("❌ Please set ROBOFLOW_API_KEY environment variable")
        print("Get your free API key at: https://roboflow.com")
        return
    
    # Find the latest processed image
    web_results = Path("web_results")
    sessions = sorted(web_results.glob("session_*"))
    if not sessions:
        print("❌ No sessions found. Please run the demo first.")
        return
    
    latest_session = sessions[-1]
    processed_images = list((latest_session / "01_pdf_processed").glob("*.png"))
    
    if not processed_images:
        print("❌ No processed images found")
        return
    
    image_path = str(processed_images[0])
    print(f"✅ Using image: {image_path}")
    
    # Initialize classifier with CubiCasa5k-2
    classifier = CubiCasaClassifier(api_key=api_key)
    
    # Create output directory
    output_dir = "cubicasa_test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Process the image
        print("🔍 Running CubiCasa5k-2 detection...")
        result = classifier.process_image(image_path, output_dir)
        
        print(f"\n✅ Detection complete!")
        print(f"📊 Detected {result['classification']['total_detections']} elements")
        print(f"🏠 Rooms: {result['analysis']['room_count']}")
        print(f"🚪 Doors: {result['analysis']['door_count']}")
        print(f"🪟 Windows: {result['analysis']['window_count']}")
        print(f"\n🖼️ Visualization saved to: {result['visualization_path']}")
        
        # Open the result
        os.system(f"open {result['visualization_path']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_cubicasa()
