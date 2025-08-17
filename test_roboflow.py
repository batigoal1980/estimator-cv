#!/usr/bin/env python3
"""
Test script for Roboflow CubiCasa5k model with proper API
"""

import os
from roboflow import Roboflow
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def test_roboflow_cubicasa():
    # Initialize Roboflow
    rf = Roboflow(api_key="ZdkZuHA7GF5NdOfCwQo8")
    
    # Try to access CubiCasa5k model
    try:
        # Try the public CubiCasa5k model first
        project = rf.workspace("floorplan-recognition").project("cubicasa5k-2-qpmsa")
        model = project.version(3).model
        print("✅ Successfully connected to CubiCasa5k-2 model!")
    except Exception as e:
        print(f"⚠️ CubiCasa5k-2 not accessible, trying alternative...")
        try:
            # Try alternative model
            project = rf.workspace("floorplan").project("cubicasa5k")
            model = project.version(1).model
            print("✅ Connected to alternative CubiCasa5k model!")
        except Exception as e2:
            print(f"❌ Error accessing models: {e2}")
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
    print(f"📁 Using image: {image_path}")
    
    # Run inference
    print("🔍 Running CubiCasa5k detection...")
    result = model.predict(image_path, confidence=50, overlap=30).json()
    
    # Process results
    predictions = result.get('predictions', [])
    print(f"\n✅ Detection complete!")
    print(f"📊 Found {len(predictions)} architectural elements")
    
    # Count by class
    class_counts = {}
    for pred in predictions:
        class_name = pred['class']
        if class_name not in class_counts:
            class_counts[class_name] = 0
        class_counts[class_name] += 1
    
    print("\n📈 Detection Summary:")
    for class_name, count in class_counts.items():
        print(f"  • {class_name}: {count}")
    
    # Visualize results
    visualize_results(image_path, predictions)
    
def visualize_results(image_path, predictions):
    """Create visualization with bounding boxes"""
    
    # Load image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.imshow(image_rgb)
    
    # Color map for different classes
    colors = {
        'Wall': 'red',
        'Door': 'blue',
        'Window': 'green',
        'Kitchen': 'orange',
        'Living Room': 'purple',
        'Bed Room': 'pink',
        'Bath': 'cyan',
        'Entry': 'yellow',
        'Storage': 'brown',
        'Garage': 'gray'
    }
    
    # Draw bounding boxes
    for pred in predictions:
        # Get box coordinates
        x = pred['x']
        y = pred['y']
        width = pred['width']
        height = pred['height']
        class_name = pred['class']
        confidence = pred['confidence']
        
        # Convert to corner coordinates
        x1 = x - width/2
        y1 = y - height/2
        
        # Get color
        color = colors.get(class_name, 'white')
        
        # Draw rectangle
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2, edgecolor=color, facecolor='none', alpha=0.8
        )
        ax.add_patch(rect)
        
        # Add label
        label = f"{class_name} ({confidence:.0f}%)"
        ax.text(x1, y1 - 5, label, 
               color=color, fontsize=8, weight='bold',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    ax.set_title(f"CubiCasa5k Detection Results - {len(predictions)} elements detected", fontsize=16)
    ax.axis('off')
    
    # Save and show
    output_path = "cubicasa_detection_result.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n🖼️ Visualization saved to: {output_path}")
    
    # Try to open the image
    os.system(f"open {output_path}")
    
    plt.show()

if __name__ == "__main__":
    test_roboflow_cubicasa()
