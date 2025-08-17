#!/usr/bin/env python3
"""Visual test to check rotation issue"""

from roboflow import Roboflow
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Initialize Roboflow
rf = Roboflow(api_key="ZdkZuHA7GF5NdOfCwQo8")
model = rf.workspace("floorplan-recognition").project("cubicasa5k-2-qpmsa").version(3).model

# Test image
test_image = "web_results/session_20250815_233455/01_pdf_processed/page_001.png"

# Load image
img = Image.open(test_image)
width, height = img.size

# Run inference
result = model.predict(test_image, confidence=25, overlap=50).json()

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 8))

# Plot 1: Original coordinates
ax1 = axes[0]
ax1.imshow(img)
ax1.set_title("Original Coordinates")
for pred in result.get('predictions', [])[:10]:  # First 10 for clarity
    x = pred['x']
    y = pred['y']
    w = pred['width']
    h = pred['height']
    
    # Convert center to corner
    x1 = x - w/2
    y1 = y - h/2
    
    rect = patches.Rectangle((x1, y1), w, h, 
                            linewidth=2, edgecolor='red', facecolor='none')
    ax1.add_patch(rect)

# Plot 2: Rotated 90° CCW (swap x,y and adjust)
ax2 = axes[1]
ax2.imshow(img)
ax2.set_title("If rotated 90° CCW")
for pred in result.get('predictions', [])[:10]:
    x = pred['x']
    y = pred['y']
    w = pred['width']
    h = pred['height']
    
    # Rotate coordinates 90° CCW
    new_x = y
    new_y = width - x
    new_w = h  # Swap width and height
    new_h = w
    
    # Convert center to corner
    x1 = new_x - new_w/2
    y1 = new_y - new_h/2
    
    rect = patches.Rectangle((x1, y1), new_w, new_h,
                            linewidth=2, edgecolor='blue', facecolor='none')
    ax2.add_patch(rect)

# Plot 3: Rotated 90° CW
ax3 = axes[2]
ax3.imshow(img)
ax3.set_title("If rotated 90° CW")
for pred in result.get('predictions', [])[:10]:
    x = pred['x']
    y = pred['y']
    w = pred['width']
    h = pred['height']
    
    # Rotate coordinates 90° CW
    new_x = height - y
    new_y = x
    new_w = h  # Swap width and height
    new_h = w
    
    # Convert center to corner
    x1 = new_x - new_w/2
    y1 = new_y - new_h/2
    
    rect = patches.Rectangle((x1, y1), new_w, new_h,
                            linewidth=2, edgecolor='green', facecolor='none')
    ax3.add_patch(rect)

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.savefig('rotation_test.png', dpi=150, bbox_inches='tight')
print("✅ Saved rotation_test.png - check which version looks correct!")
plt.show()
