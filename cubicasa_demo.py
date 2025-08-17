#!/usr/bin/env python3
"""
Roboflow CubiCasa5k Integration for Estimator CV Demo
Floor-plan-aware classification of architectural elements
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
# import cv2  # Removed to avoid libGL dependency on Railway
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import logging
import json
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CubiCasaClassifier:
    """Floor plan classification using Roboflow's CubiCasa5k model"""
    
    def __init__(self, api_key: str = None, model_url: str = None):
        """
        Initialize CubiCasa classifier
        
        Args:
            api_key: Roboflow API key (will try to get from environment if None)
            model_url: Custom model URL (will use default CubiCasa5k if None)
        """
        self.api_key = api_key or os.getenv('ROBOFLOW_API_KEY')
        if not self.api_key:
            logger.warning("No Roboflow API key provided. Set ROBOFLOW_API_KEY environment variable.")
        
        # CubiCasa5k model - use the public version
        # Try different model versions based on what's accessible
        self.model_url = model_url or "https://detect.roboflow.com/cubicasa5k/1"
        
        # CubiCasa5k-2 model class names (actual floor plan elements)
        self.class_names = {
            "Background": 0,
            "Outdoor": 1, 
            "Wall": 2,
            "Kitchen": 3,
            "Living Room": 4,
            "Bed Room": 5,
            "Bath": 6,
            "Entry": 7,
            "Railing": 8,
            "Storage": 9,
            "Garage": 10,
            "Undefined": 11,
            "Door": 12,
            "Window": 13
        }
        
        # Color mapping for visualization - distinct colors for each class
        self.class_colors = {
            'wall': '#0000FF',      # Blue
            'door': '#FF0000',      # Red
            'window': '#FFFF00',    # Yellow
            'room': '#00FF00',      # Green
            'stairs': '#FF00FF',    # Magenta
            'symbol': '#00FFFF',    # Cyan
            'text': '#FFA500',      # Orange
            'furniture': '#800080', # Purple
            'appliance': '#008000', # Dark Green
            'fixture': '#000080',   # Navy
            'other': '#808080'      # Gray
        }
        
        logger.info("CubiCasa classifier initialized")
    
    def classify_image(self, image_path: str, confidence_threshold: float = 0.25) -> Dict[str, Any]:
        """
        Classify architectural elements in an image using Roboflow API
        
        Args:
            image_path: Path to input image
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            Dictionary with classification results
        """
        if not self.api_key:
            raise RuntimeError("Roboflow API key not available")
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        logger.info(f"Classifying image: {image_path}")
        
        # Store confidence for visualization
        self.last_confidence = confidence_threshold
        
        # Check if we're on Railway and SDK might have libGL issues
        on_railway = bool(os.getenv('RAILWAY_ENVIRONMENT'))
        
        if not on_railway:
            try:
                # Use the Roboflow Python SDK (local only)
                from roboflow import Roboflow
                
                # Initialize Roboflow
                rf = Roboflow(api_key=self.api_key)
                
                # Access CubiCasa5k-2 model (version 3)
                project = rf.workspace("floorplan-recognition").project("cubicasa5k-2-qpmsa")
                model = project.version(3).model
                
                # Run inference
                confidence_int = int(confidence_threshold * 100)
                logger.info(f"🎯 SDK API Parameters: confidence={confidence_int}, overlap=50")
                result = model.predict(image_path, confidence=confidence_int, overlap=50).json()
                
            except Exception as e:
                logger.warning(f"SDK failed locally: {e}, trying REST API...")
                on_railway = True  # Fall through to REST API
        
        if on_railway:
            # Use REST API directly on Railway (avoids libGL issues)
            logger.info("Using Roboflow REST API (Railway/headless mode)")
            result = self._call_rest_api(image_path, confidence_threshold)
        
        # Process predictions (works for both SDK and REST API results)
        detections = []
        for prediction in result.get('predictions', []):
            detection = {
                'class': prediction['class'],
                'confidence': float(prediction['confidence']),  # Ensure it's a float
                'bbox': {
                    'x': float(prediction['x']),
                    'y': float(prediction['y']),
                    'width': float(prediction['width']),
                    'height': float(prediction['height'])
                }
            }
            detections.append(detection)
        
        # Get image dimensions from API response (these are the dimensions used for predictions)
        api_width = result.get('image', {}).get('width')
        api_height = result.get('image', {}).get('height')
        
        # Also get actual image dimensions for scaling if needed
        from PIL import Image
        with Image.open(image_path) as img:
            actual_width, actual_height = img.size
        
        # Use API dimensions if available, otherwise use actual dimensions
        # Ensure dimensions are integers
        img_width = int(api_width) if api_width else actual_width
        img_height = int(api_height) if api_height else actual_height
        
        result_data = {
            'image_path': image_path,
            'image_width': img_width,
            'image_height': img_height,
            'actual_width': actual_width,
            'actual_height': actual_height,
            'detections': detections,
            'total_detections': len(detections)
        }
        
        logger.info(f"✅ Detected {len(detections)} architectural elements using {'REST API' if on_railway else 'SDK'}")
        return result_data
    
    def visualize_classification(self, image_path: str, classification_result: Dict[str, Any], 
                                output_path: str = None, use_sdk_viz: bool = False) -> str:
        """
        Visualize classification results
        
        Args:
            image_path: Path to input image
            classification_result: Classification results from classify_image
            output_path: Path to save visualization
            use_sdk_viz: If True, use Roboflow SDK's built-in visualization
            
        Returns:
            Path to saved visualization
        """
        
        # If SDK visualization is requested and available, use it
        if use_sdk_viz:
            try:
                from roboflow import Roboflow
                
                # Re-run prediction to get the result object
                rf = Roboflow(api_key=self.api_key)
                model = rf.workspace("floorplan-recognition").project("cubicasa5k-2-qpmsa").version(3).model
                
                # Get confidence threshold from the last run or use default
                confidence = int(self.last_confidence * 100) if hasattr(self, 'last_confidence') else 25
                
                # Run prediction
                result = model.predict(image_path, confidence=confidence, overlap=50)
                result_json = result.json()
                
                logger.info(f"SDK detected {len(result_json.get('predictions', []))} elements with confidence {confidence}%")
                
                # Count detections by class for logging
                class_counts = {}
                for pred in result_json.get('predictions', []):
                    cls = pred['class']
                    class_counts[cls] = class_counts.get(cls, 0) + 1
                
                logger.info(f"Detection breakdown: {class_counts}")
                
                # Save using SDK's built-in visualization
                if output_path is None:
                    output_path = image_path.replace('.png', '_cubicasa_demo.png')
                
                result.save(output_path)
                logger.info(f"✅ Saved SDK visualization to {output_path}")
                return output_path
                
            except Exception as e:
                logger.warning(f"Supervision visualization failed: {e}")
                logger.info("Falling back to SDK's default visualization")
                
                try:
                    # Fallback to SDK's built-in save method
                    from roboflow import Roboflow
                    rf = Roboflow(api_key=self.api_key)
                    model = rf.workspace("floorplan-recognition").project("cubicasa5k-2-qpmsa").version(3).model
                    confidence = int(self.last_confidence * 100) if hasattr(self, 'last_confidence') else 25
                    
                    result = model.predict(image_path, confidence=confidence, overlap=50)
                    
                    if output_path is None:
                        output_path = image_path.replace('.png', '_cubicasa_demo.png')
                    
                    # Use SDK's built-in visualization (will be all blue but has all detections)
                    result.save(output_path)
                    
                    predictions = result.json().get('predictions', [])
                    logger.info(f"✅ SDK default visualization saved with {len(predictions)} detections")
                    return output_path
                    
                except Exception as e2:
                    logger.error(f"SDK default visualization also failed: {e2}")
                    # Fall through to custom visualization
        
        # Set matplotlib to non-interactive backend to avoid GUI issues
        import matplotlib
        matplotlib.use('Agg')
        
        # Load image using PIL instead of cv2 (to avoid libGL dependency on Railway)
        pil_image = Image.open(image_path)
        if pil_image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Convert to RGB and ensure uint8 format (same as cv2 would produce)
        image_rgb = np.array(pil_image.convert('RGB'), dtype=np.uint8)
        h, w = image_rgb.shape[:2]
        
        # Debug info for comparison
        logger.info(f"📐 Image dimensions: {w}x{h}, dtype: {image_rgb.dtype}")
        logger.info(f"🎨 Pixel range: [{image_rgb.min()}-{image_rgb.max()}]")
        
        # Get API dimensions for scaling
        api_width = classification_result.get('image_width', w)
        api_height = classification_result.get('image_height', h)
        
        # Calculate scaling factors if API used different dimensions
        scale_x = w / api_width if api_width and api_width != w else 1.0
        scale_y = h / api_height if api_height and api_height != h else 1.0
        
        if scale_x != 1.0 or scale_y != 1.0:
            logger.info(f"Scaling coordinates: API {api_width}x{api_height} -> Actual {w}x{h}")
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Original image
        ax1.imshow(image_rgb)
        ax1.set_title("Original Image", fontsize=14)
        ax1.axis('off')
        
        # Annotated image
        ax2.imshow(image_rgb)
        
        # Draw bounding boxes and labels
        detection_count = {}
        for detection in classification_result['detections']:
            class_name = detection['class']
            confidence = detection['confidence']
            bbox = detection['bbox']
            
            # Count detections by class
            if class_name not in detection_count:
                detection_count[class_name] = 0
            detection_count[class_name] += 1
            
            # Scale coordinates to match actual image dimensions
            x_center = bbox['x'] * scale_x
            y_center = bbox['y'] * scale_y
            width = bbox['width'] * scale_x
            height = bbox['height'] * scale_y
            
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            
            # Debug coordinate scaling
            if scale_x != 1.0 or scale_y != 1.0:
                logger.debug(f"Scaled bbox: ({bbox['x']}, {bbox['y']}) -> ({x_center:.1f}, {y_center:.1f})")
            
            # Get color for this class
            color = self.class_colors.get(class_name.lower(), '#808080')  # Default to gray if class not found
            
            # Debug: log the class name and color mapping
            logger.debug(f"Class: '{class_name}' -> '{class_name.lower()}' -> Color: {color}")
            
            # Draw bounding box with class-specific color
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2, edgecolor=color, facecolor='none', alpha=0.8
            )
            ax2.add_patch(rect)
            
            # Add label with matching color
            label = f"{class_name} ({confidence:.2f})"
            ax2.text(x1, y1 - 5, label, 
                    color=color, fontsize=8, weight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        ax2.set_title(f"Classification Results - {len(classification_result['detections'])} detections", fontsize=14)
        ax2.axis('off')
        
        # Add legend with color coding
        from matplotlib.patches import Patch
        legend_elements = []
        for class_name, count in sorted(detection_count.items()):
            color = self.class_colors.get(class_name.lower(), '#808080')
            legend_elements.append(Patch(facecolor=color, label=f'{class_name}: {count}'))
        
        if legend_elements:
            ax2.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        # Add detection summary
        summary_text = "Detection Summary:\n"
        for class_name, count in detection_count.items():
            summary_text += f"• {class_name}: {count}\n"
        
        plt.figtext(0.02, 0.02, summary_text, fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
        
        plt.suptitle(f"CubiCasa5k Floor Plan Classification - {Path(image_path).name}", fontsize=16)
        plt.tight_layout()
        
        # Save visualization
        if output_path is None:
            output_path = image_path.replace('.png', '_cubicasa_demo.png')
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Classification visualization saved to: {output_path}")
        return output_path
    
    def analyze_floor_plan(self, classification_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze floor plan based on classification results
        
        Args:
            classification_result: Classification results from classify_image
            
        Returns:
            Dictionary with floor plan analysis
        """
        detections = classification_result['detections']
        
        # Count by class
        class_counts = {}
        for detection in detections:
            class_name = detection['class']
            if class_name not in class_counts:
                class_counts[class_name] = 0
            class_counts[class_name] += 1
        
        # Calculate areas
        total_area = classification_result['image_width'] * classification_result['image_height']
        room_area = 0
        wall_area = 0
        
        for detection in detections:
            bbox = detection['bbox']
            area = (bbox['width'] * bbox['height']) / 10000  # Convert to percentage
            
            if detection['class'] == 'room':
                room_area += area
            elif detection['class'] == 'wall':
                wall_area += area
        
        # Floor plan analysis
        analysis = {
            'total_elements': len(detections),
            'class_distribution': class_counts,
            'room_count': class_counts.get('room', 0),
            'door_count': class_counts.get('door', 0),
            'window_count': class_counts.get('window', 0),
            'wall_count': class_counts.get('wall', 0),
            'room_area_percentage': room_area,
            'wall_area_percentage': wall_area,
            'openness_ratio': room_area / (room_area + wall_area) if (room_area + wall_area) > 0 else 0
        }
        
        return analysis
    
    def process_image(self, image_path: str, output_dir: str, 
                     confidence_threshold: float = 0.25) -> Dict[str, Any]:
        """
        Complete classification pipeline for a single image
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save outputs
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            Dictionary with classification results and analysis
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Classify image
        classification_result = self.classify_image(image_path, confidence_threshold)
        
        # Analyze floor plan
        analysis = self.analyze_floor_plan(classification_result)
        
        # Create visualization
        viz_path = self.visualize_classification(
            image_path, 
            classification_result, 
            os.path.join(output_dir, f"{Path(image_path).stem}_cubicasa_demo.png")
        )
        
        # Save results
        results = {
            'classification': classification_result,
            'analysis': analysis,
            'visualization_path': viz_path
        }
        
        # Save to JSON
        json_path = os.path.join(output_dir, f"{Path(image_path).stem}_cubicasa_results.json")
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to: {json_path}")
        
        return results
    
    def process_directory(self, input_dir: str, output_dir: str, 
                         confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Process all images in a directory
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save outputs
            confidence_threshold: Minimum confidence for detections
            
        Returns:
            List of results for each image
        """
        image_extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(Path(input_dir).glob(f"*{ext}"))
            image_files.extend(Path(input_dir).glob(f"*{ext.upper()}"))
        
        if not image_files:
            logger.warning(f"No image files found in {input_dir}")
            return []
        
        logger.info(f"Found {len(image_files)} image files to process")
        
        results = []
        for image_file in image_files:
            try:
                result = self.process_image(str(image_file), output_dir, confidence_threshold)
                results.append(result)
                logger.info(f"Processed: {image_file.name}")
            except Exception as e:
                logger.error(f"Error processing {image_file}: {e}")
                continue
        
        return results
    
    def _call_rest_api(self, image_path: str, confidence_threshold: float) -> Dict[str, Any]:
        """
        Call Roboflow REST API directly (avoids SDK libGL issues)
        """
        import requests
        import base64
        
        # Read and encode image
        with open(image_path, 'rb') as image_file:
            image_data = base64.b64encode(image_file.read()).decode()
        
        # Roboflow API endpoint for CubiCasa5k-2 v3
        url = f"https://detect.roboflow.com/cubicasa5k-2-qpmsa/3"
        
        # API parameters
        params = {
            'api_key': self.api_key,
            'confidence': int(confidence_threshold * 100),
            'overlap': 50
        }
        
        # Make POST request
        logger.info(f"🌐 REST API call: confidence={params['confidence']}, overlap={params['overlap']}")
        response = requests.post(url, 
                               data=image_data,
                               params=params,
                               headers={'Content-Type': 'application/x-www-form-urlencoded'})
        
        if response.status_code != 200:
            raise Exception(f"API request failed: {response.status_code} - {response.text}")
        
        result = response.json()
        logger.info(f"✅ REST API returned {len(result.get('predictions', []))} detections")
        return result
    
    def demo_without_api(self, image_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Demo mode that shows the interface without requiring API key
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save outputs
            
        Returns:
            Mock results for demonstration
        """
        logger.info("Running in demo mode (no API key required)")
        
        # Create mock classification result
        mock_result = {
            'image_path': image_path,
            'image_width': 800,
            'image_height': 600,
            'detections': [
                {
                    'class': 'room',
                    'confidence': 0.95,
                    'bbox': {'x': 25, 'y': 30, 'width': 40, 'height': 35}
                },
                {
                    'class': 'door',
                    'confidence': 0.87,
                    'bbox': {'x': 45, 'y': 60, 'width': 8, 'height': 15}
                },
                {
                    'class': 'window',
                    'confidence': 0.92,
                    'bbox': {'x': 15, 'y': 20, 'width': 12, 'height': 8}
                }
            ],
            'total_detections': 3
        }
        
        # Create visualization
        viz_path = self.visualize_classification(
            image_path, 
            mock_result, 
            os.path.join(output_dir, f"{Path(image_path).stem}_cubicasa_demo.png")
        )
        
        # Mock analysis
        analysis = {
            'total_elements': 3,
            'class_distribution': {'room': 1, 'door': 1, 'window': 1},
            'room_count': 1,
            'door_count': 1,
            'window_count': 1,
            'wall_count': 0,
            'room_area_percentage': 14.0,
            'wall_area_percentage': 0.0,
            'openness_ratio': 1.0
        }
        
        return {
            'classification': mock_result,
            'analysis': analysis,
            'visualization_path': viz_path,
            'demo_mode': True
        }

def main():
    """Command line interface for CubiCasa classification"""
    parser = argparse.ArgumentParser(description="CubiCasa5k Floor Plan Classification for Estimator CV Demo")
    parser.add_argument("--input", "-i", required=True, help="Input image file or directory")
    parser.add_argument("--output", "-o", required=True, help="Output directory for results")
    parser.add_argument("--confidence", "-c", type=float, default=0.5, help="Confidence threshold (default: 0.5)")
    parser.add_argument("--api-key", help="Roboflow API key (or set ROBOFLOW_API_KEY env var)")
    parser.add_argument("--demo", action="store_true", help="Run in demo mode without API key")
    
    args = parser.parse_args()
    
    try:
        classifier = CubiCasaClassifier(api_key=args.api_key)
        
        if args.demo or not classifier.api_key:
            logger.info("Running in demo mode")
            if os.path.isfile(args.input):
                result = classifier.demo_without_api(args.input, args.output)
                print(f"\n✅ Demo classification complete!")
                print(f"📊 Mock detections: {result['classification']['total_detections']}")
                print(f"🖼️  Visualization: {result['visualization_path']}")
            else:
                print("Demo mode only supports single image files")
        else:
            if os.path.isfile(args.input):
                # Process single image
                result = classifier.process_image(args.input, args.output, args.confidence)
                print(f"\n✅ Classification complete!")
                print(f"📊 Detected {result['classification']['total_detections']} elements")
                print(f"🖼️  Visualization: {result['visualization_path']}")
                print(f"🏠 Room count: {result['analysis']['room_count']}")
                print(f"🚪 Door count: {result['analysis']['door_count']}")
                print(f"🪟 Window count: {result['analysis']['window_count']}")
                
            elif os.path.isdir(args.input):
                # Process directory
                results = classifier.process_directory(args.input, args.output, args.confidence)
                print(f"\n✅ Classification complete!")
                print(f"📁 Processed {len(results)} images")
                
            else:
                print(f"Error: {args.input} is not a valid file or directory")
                sys.exit(1)
                
    except Exception as e:
        logger.error(f"Classification failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
