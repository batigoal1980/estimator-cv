#!/usr/bin/env python3
"""
Layout Detector for Estimator CV Demo
Uses LayoutParser to detect and visualize document regions
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import logging

# LayoutParser imports
try:
    import layoutparser as lp
    LAYOUTPARSER_AVAILABLE = True
    print("LayoutParser is installed successfully!")
except ImportError:
    print("Warning: LayoutParser not installed. Layout detection will be disabled.")
    print("To enable, install with: pip install layoutparser[ocr]")
    LAYOUTPARSER_AVAILABLE = False
    lp = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LayoutDetector:
    """Layout detection using LayoutParser for architectural drawings"""
    
    def __init__(self, model_name: str = "PubLayNet"):
        """
        Initialize layout detector
        
        Args:
            model_name: Name of the layout detection model
        """
        self.model_name = model_name
        self.model = None
        self.ocr_agent = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize layout detection and OCR models"""
        if not LAYOUTPARSER_AVAILABLE:
            logger.error("LayoutParser not available - LAYOUTPARSER_AVAILABLE is False")
            self.model = None
            self.ocr_agent = None
            return
            
        try:
            logger.info(f"LayoutParser version: {lp.__version__ if hasattr(lp, '__version__') else 'unknown'}")
            logger.info(f"LayoutParser attributes: {dir(lp)[:10]}...")  # Show first 10 attributes
            logger.info(f"Initializing {self.model_name} model...")
            
            # Try different model backends
            try:
                # Try Detectron2 backend first
                logger.info("Attempting to load Detectron2LayoutModel...")
                if hasattr(lp, 'Detectron2LayoutModel'):
                    self.model = lp.Detectron2LayoutModel(
                        config_path='lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
                        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
                        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5]
                    )
                    logger.info("✅ Successfully loaded Detectron2 backend")
                else:
                    logger.warning("Detectron2LayoutModel not found in layoutparser")
                    raise AttributeError("Detectron2LayoutModel not available")
            except Exception as e:
                logger.warning(f"Detectron2 backend failed: {e}")
                try:
                    # Try PaddleDetection backend
                    logger.info("Attempting to load PaddleDetectionLayoutModel...")
                    if hasattr(lp, 'PaddleDetectionLayoutModel'):
                        self.model = lp.PaddleDetectionLayoutModel(
                            config_path='lp://PubLayNet/ppyolov2_r50vd_dcn_365e',
                            label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
                            threshold=0.5,
                            enable_mkldnn=True
                        )
                        logger.info("✅ Successfully loaded PaddleDetection backend")
                    else:
                        logger.warning("PaddleDetectionLayoutModel not found in layoutparser")
                        raise AttributeError("PaddleDetectionLayoutModel not available")
                except Exception as e2:
                    logger.error(f"PaddleDetection backend also failed: {e2}")
                    logger.error("No layout detection models available")
                    self.model = None
            
            # Initialize OCR agent for text extraction
            try:
                self.ocr_agent = lp.TesseractAgent(languages=['eng'])
            except:
                logger.warning("Tesseract not available, OCR disabled")
                self.ocr_agent = None
            
            if self.model:
                logger.info("Models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            logger.warning("Will run in demo mode without actual layout detection")
            self.model = None
            self.ocr_agent = None
    
    def detect_layout(self, image_path: str) -> List[Any]:
        """
        Detect layout regions in an image
        
        Args:
            image_path: Path to input image
            
        Returns:
            List of detected layout regions
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        logger.info(f"Detecting layout in: {image_path}")
        
        try:
            # Load image
            if LAYOUTPARSER_AVAILABLE and hasattr(lp, 'load_image'):
                image = lp.load_image(image_path)
            else:
                # Fallback to PIL/cv2
                from PIL import Image
                import numpy as np
                image = np.array(Image.open(image_path))
            
            # Detect layout
            if self.model is not None:
                logger.info(f"Model type: {type(self.model)}")
                logger.info("Attempting to detect layout with model...")
                layout = self.model.detect(image)
                logger.info(f"✅ Layout detection successful! Found {len(layout)} regions")
            else:
                logger.warning("Model is None - using demo mode for layout detection")
                logger.info("To enable real detection, install: pip install paddlepaddle paddledet")
                # Return mock layout for demo purposes
                layout = self._create_mock_layout_simple()
            
            # Extract text from text regions (only if we have real layout)
            if self.model is not None and self.ocr_agent is not None:
                for block in layout:
                    if hasattr(block, '__class__') and block.__class__.__name__ == 'TextBlock':
                        try:
                            block.set(text=self.ocr_agent.detect(block.crop(image)))
                        except:
                            pass
            
            logger.info(f"Detected {len(layout)} layout regions")
            return layout
            
        except Exception as e:
            logger.error(f"Error detecting layout: {e}")
            raise
    
    def visualize_layout(self, image_path: str, layout: List[Any], output_path: str = None) -> str:
        """
        Visualize detected layout regions
        
        Args:
            image_path: Path to input image
            layout: Detected layout regions
            output_path: Path to save visualization (optional)
            
        Returns:
            Path to saved visualization
        """
        # Load image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Use non-interactive backend to avoid GUI issues
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        ax.imshow(image_rgb)
        
        # Color mapping for different region types
        colors = {
            'Text': 'blue',
            'Title': 'red', 
            'List': 'green',
            'Table': 'orange',
            'Figure': 'purple'
        }
        
        # Draw bounding boxes
        for block in layout:
            # Get block coordinates
            x1, y1, x2, y2 = block.block.coordinates
            
            # Get block type
            block_type = block.type if hasattr(block, 'type') else 'Unknown'
            color = colors.get(block_type, 'gray')
            
            # Create rectangle patch
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor='none', alpha=0.8
            )
            ax.add_patch(rect)
            
            # Add label
            ax.text(x1, y1 - 5, f"{block_type}", 
                   color=color, fontsize=10, weight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7))
            
            # Add text preview for text blocks
            if hasattr(block, 'text') and block.text:
                text_preview = block.text[:50] + "..." if len(block.text) > 50 else block.text
                ax.text(x1, y2 + 15, text_preview, 
                       color='black', fontsize=8, 
                       bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        
        ax.set_title(f"Layout Detection Results - {os.path.basename(image_path)}", fontsize=16)
        ax.axis('off')
        
        # Save visualization
        if output_path is None:
            output_path = image_path.replace('.png', '_layout.png')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Layout visualization saved to: {output_path}")
        return output_path
    
    def _create_mock_layout_simple(self) -> List:
        """Create simple mock layout for demo purposes"""
        # Return empty list for now - the demo will continue
        return []
    
    def analyze_regions(self, layout: List[Any]) -> Dict[str, Any]:
        """
        Analyze detected layout regions
        
        Args:
            layout: Detected layout regions
            
        Returns:
            Dictionary with region statistics
        """
        stats = {
            'total_regions': len(layout),
            'region_types': {},
            'text_regions': 0,
            'figure_regions': 0,
            'table_regions': 0
        }
        
        for block in layout:
            block_type = block.type if hasattr(block, 'type') else 'Unknown'
            
            # Count region types
            if block_type not in stats['region_types']:
                stats['region_types'][block_type] = 0
            stats['region_types'][block_type] += 1
            
            # Count specific types
            if block_type == 'Text':
                stats['text_regions'] += 1
            elif block_type == 'Figure':
                stats['figure_regions'] += 1
            elif block_type == 'Table':
                stats['table_regions'] += 1
        
        return stats
    
    def process_image(self, image_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Complete layout detection pipeline for a single image
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary with results and file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Detect layout
        layout = self.detect_layout(image_path)
        
        # Analyze regions
        stats = self.analyze_regions(layout)
        
        # Create visualization
        viz_path = self.visualize_layout(
            image_path, 
            layout, 
            os.path.join(output_dir, f"{Path(image_path).stem}_layout.png")
        )
        
        # Save layout data
        layout_data = {
            'image_path': image_path,
            'layout': [block.to_dict() if hasattr(block, 'to_dict') else str(block) for block in layout],
            'statistics': stats
        }
        
        return {
            'layout': layout,
            'statistics': stats,
            'visualization_path': viz_path,
            'layout_data': layout_data
        }
    
    def process_directory(self, input_dir: str, output_dir: str) -> List[Dict[str, Any]]:
        """
        Process all images in a directory
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save outputs
            
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
                result = self.process_image(str(image_file), output_dir)
                results.append(result)
                logger.info(f"Processed: {image_file.name}")
            except Exception as e:
                logger.error(f"Error processing {image_file}: {e}")
                continue
        
        return results

def main():
    """Command line interface for layout detection"""
    parser = argparse.ArgumentParser(description="Layout Detector for Estimator CV Demo")
    parser.add_argument("--input", "-i", required=True, help="Input image file or directory")
    parser.add_argument("--output", "-o", required=True, help="Output directory for results")
    parser.add_argument("--model", default="PubLayNet", help="Layout detection model (default: PubLayNet)")
    
    args = parser.parse_args()
    
    try:
        detector = LayoutDetector(model_name=args.model)
        
        if os.path.isfile(args.input):
            # Process single image
            result = detector.process_image(args.input, args.output)
            print(f"\n✅ Layout detection complete!")
            print(f"📊 Detected {result['statistics']['total_regions']} regions")
            print(f"🖼️  Visualization: {result['visualization_path']}")
            
        elif os.path.isdir(args.input):
            # Process directory
            results = detector.process_directory(args.input, args.output)
            print(f"\n✅ Layout detection complete!")
            print(f"📁 Processed {len(results)} images")
            print(f"📊 Total regions detected: {sum(r['statistics']['total_regions'] for r in results)}")
            
        else:
            print(f"Error: {args.input} is not a valid file or directory")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Layout detection failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
