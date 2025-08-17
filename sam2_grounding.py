#!/usr/bin/env python3
"""
SAM2 + Grounding-DINO Integration for Estimator CV Demo
Interactive segmentation of architectural elements
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import logging
import json

# Try to import SAM2 and Grounding-DINO
try:
    import torch
    from segment_anything_2 import sam_model_registry, SamPredictor
    from groundingdino.util.inference import load_model, load_image, predict, annotate
    from groundingdino.util.utils import clean_state_dict
    SAM2_AVAILABLE = True
except ImportError as e:
    print(f"Warning: SAM2/Grounding-DINO not installed: {e}")
    print("Segmentation will run in demo mode.")
    SAM2_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SAM2GroundingSegmenter:
    """Interactive segmentation using SAM2 and Grounding-DINO"""
    
    def __init__(self, sam2_checkpoint: str = None, grounding_dino_config: str = None):
        """
        Initialize SAM2 and Grounding-DINO models
        
        Args:
            sam2_checkpoint: Path to SAM2 checkpoint (will download if None)
            grounding_dino_config: Path to Grounding-DINO config (will download if None)
        """
        self.sam2_predictor = None
        self.grounding_dino_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Using device: {self.device}")
        self._initialize_models(sam2_checkpoint, grounding_dino_config)
    
    def _initialize_models(self, sam2_checkpoint: str, grounding_dino_config: str):
        """Initialize both SAM2 and Grounding-DINO models"""
        try:
            # Initialize SAM2
            logger.info("Initializing SAM2...")
            if sam2_checkpoint is None:
                # Use default SAM2 model
                sam2_checkpoint = "sam2_hq.pth"
                if not os.path.exists(sam2_checkpoint):
                    logger.info("Downloading SAM2 checkpoint...")
                    # This would download the model - in practice, you'd use the actual download URL
                    logger.warning("Please download SAM2 checkpoint manually and place in current directory")
            
            # Initialize SAM2 model
            sam2_model = sam_model_registry["vit_h"](checkpoint=sam2_checkpoint)
            sam2_model.to(device=self.device)
            self.sam2_predictor = SamPredictor(sam2_model)
            
            # Initialize Grounding-DINO
            logger.info("Initializing Grounding-DINO...")
            if grounding_dino_config is None:
                # Use default Grounding-DINO model
                grounding_dino_config = "groundingdino_swint_ogc.py"
                grounding_dino_checkpoint = "groundingdino_swint_ogc.pth"
                
                if not os.path.exists(grounding_dino_checkpoint):
                    logger.info("Downloading Grounding-DINO checkpoint...")
                    # This would download the model - in practice, you'd use the actual download URL
                    logger.warning("Please download Grounding-DINO checkpoint manually and place in current directory")
            
            # Initialize Grounding-DINO model
            self.grounding_dino_model = load_model(grounding_dino_config, grounding_dino_checkpoint)
            self.grounding_dino_model.to(device=self.device)
            
            logger.info("Models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise
    
    def segment_by_click(self, image_path: str, click_points: List[Tuple[int, int]], 
                         click_labels: List[int] = None) -> np.ndarray:
        """
        Segment image using SAM2 with click points
        
        Args:
            image_path: Path to input image
            click_points: List of (x, y) click coordinates
            click_labels: List of click labels (1 for positive, 0 for negative)
            
        Returns:
            Segmentation mask
        """
        if self.sam2_predictor is None:
            raise RuntimeError("SAM2 model not initialized")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Set image in SAM2 predictor
        self.sam2_predictor.set_image(image)
        
        # Prepare input points and labels
        if click_labels is None:
            click_labels = [1] * len(click_points)  # Default to positive clicks
        
        input_points = np.array(click_points)
        input_labels = np.array(click_labels)
        
        # Generate mask
        masks, scores, logits = self.sam2_predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True
        )
        
        # Select best mask (highest score)
        best_mask_idx = np.argmax(scores)
        best_mask = masks[best_mask_idx]
        
        logger.info(f"Generated mask with score: {scores[best_mask_idx]:.3f}")
        return best_mask
    
    def segment_by_text(self, image_path: str, text_prompt: str, 
                        confidence_threshold: float = 0.35) -> Tuple[np.ndarray, Dict]:
        """
        Segment image using Grounding-DINO with text prompt
        
        Args:
            image_path: Path to input image
            text_prompt: Text description of what to segment
            confidence_threshold: Minimum confidence for detection
            
        Returns:
            Tuple of (segmentation_mask, detection_info)
        """
        if self.grounding_dino_model is None:
            raise RuntimeError("Grounding-DINO model not initialized")
        
        # Load image
        image_source, image = load_image(image_path)
        
        # Run Grounding-DINO detection
        boxes, logits, phrases = predict(
            model=self.grounding_dino_model,
            image=image_source,
            caption=text_prompt,
            box_threshold=confidence_threshold,
            text_threshold=confidence_threshold
        )
        
        if len(boxes) == 0:
            logger.warning(f"No objects detected for prompt: '{text_prompt}'")
            return np.zeros((image.shape[0], image.shape[1]), dtype=bool), {}
        
        # Get the first (highest confidence) detection
        box = boxes[0]
        confidence = logits[0]
        phrase = phrases[0]
        
        logger.info(f"Detected '{phrase}' with confidence: {confidence:.3f}")
        
        # Convert box coordinates to image coordinates
        h, w = image.shape[:2]
        x1, y1, x2, y2 = box.cpu().numpy()
        x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
        
        # Create bounding box mask
        mask = np.zeros((h, w), dtype=bool)
        mask[y1:y2, x1:x2] = True
        
        detection_info = {
            'box': [x1, y1, x2, y2],
            'confidence': float(confidence),
            'phrase': phrase
        }
        
        return mask, detection_info
    
    def refine_with_sam2(self, image_path: str, initial_mask: np.ndarray, 
                         refinement_points: List[Tuple[int, int]] = None) -> np.ndarray:
        """
        Refine segmentation mask using SAM2
        
        Args:
            image_path: Path to input image
            initial_mask: Initial segmentation mask
            refinement_points: Additional click points for refinement
            
        Returns:
            Refined segmentation mask
        """
        if self.sam2_predictor is None:
            raise RuntimeError("SAM2 model not initialized")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Set image in SAM2 predictor
        self.sam2_predictor.set_image(image)
        
        if refinement_points:
            # Use refinement points to improve mask
            input_points = np.array(refinement_points)
            input_labels = np.array([1] * len(refinement_points))  # Positive clicks
            
            masks, scores, logits = self.sam2_predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                mask_input=initial_mask.astype(np.uint8),
                multimask_output=False
            )
            
            refined_mask = masks[0]
        else:
            # Use initial mask as is
            refined_mask = initial_mask
        
        return refined_mask
    
    def visualize_segmentation(self, image_path: str, mask: np.ndarray, 
                              output_path: str = None, title: str = "Segmentation Result") -> str:
        """
        Visualize segmentation results
        
        Args:
            image_path: Path to input image
            mask: Segmentation mask
            output_path: Path to save visualization
            title: Title for the visualization
            
        Returns:
            Path to saved visualization
        """
        # Load image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create figure with subplots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        ax1.imshow(image_rgb)
        ax1.set_title("Original Image", fontsize=14)
        ax1.axis('off')
        
        # Mask
        ax2.imshow(mask, cmap='gray')
        ax2.set_title("Segmentation Mask", fontsize=14)
        ax2.axis('off')
        
        # Overlay
        overlay = image_rgb.copy()
        overlay[mask] = [255, 0, 0]  # Red overlay for segmented regions
        ax3.imshow(overlay)
        ax3.set_title("Overlay", fontsize=14)
        ax3.axis('off')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        # Save visualization
        if output_path is None:
            output_path = image_path.replace('.png', '_segmentation.png')
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Segmentation visualization saved to: {output_path}")
        return output_path
    
    def interactive_segmentation(self, image_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Interactive segmentation demo
        
        Args:
            image_path: Path to input image
            output_dir: Directory to save outputs
            
        Returns:
            Dictionary with segmentation results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Example text prompts for architectural elements
        architectural_prompts = [
            "room", "door", "window", "wall", "floor", "ceiling",
            "electrical outlet", "light fixture", "furniture", "cabinet",
            "counter", "sink", "toilet", "bathtub", "shower"
        ]
        
        results = {}
        
        # Text-based segmentation examples
        logger.info("Running text-based segmentation examples...")
        for prompt in architectural_prompts[:3]:  # Test first 3 prompts
            try:
                mask, detection_info = self.segment_by_text(image_path, prompt)
                
                if mask.any():  # If anything was detected
                    # Visualize result
                    viz_path = self.visualize_segmentation(
                        image_path, mask,
                        os.path.join(output_dir, f"{Path(image_path).stem}_{prompt}_seg.png"),
                        f"Text Prompt: '{prompt}'"
                    )
                    
                    results[prompt] = {
                        'mask': mask,
                        'detection_info': detection_info,
                        'visualization_path': viz_path
                    }
                    
                    logger.info(f"Successfully segmented '{prompt}'")
                
            except Exception as e:
                logger.warning(f"Failed to segment '{prompt}': {e}")
                continue
        
        # Click-based segmentation example (center of image)
        logger.info("Running click-based segmentation example...")
        try:
            # Get image dimensions
            image = cv2.imread(image_path)
            h, w = image.shape[:2]
            
            # Click at center of image
            center_point = [(w // 2, h // 2)]
            click_mask = self.segment_by_click(image_path, center_point)
            
            if click_mask.any():
                viz_path = self.visualize_segmentation(
                    image_path, click_mask,
                    os.path.join(output_dir, f"{Path(image_path).stem}_click_seg.png"),
                    "Click-based Segmentation (Center)"
                )
                
                results['click_center'] = {
                    'mask': click_mask,
                    'click_points': center_point,
                    'visualization_path': viz_path
                }
                
                logger.info("Successfully generated click-based segmentation")
        
        except Exception as e:
            logger.warning(f"Failed to generate click-based segmentation: {e}")
        
        return results
    
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
                result = self.interactive_segmentation(str(image_file), output_dir)
                results.append({
                    'image_file': str(image_file),
                    'segmentations': result
                })
                logger.info(f"Processed: {image_file.name}")
            except Exception as e:
                logger.error(f"Error processing {image_file}: {e}")
                continue
        
        return results

def main():
    """Command line interface for SAM2 + Grounding-DINO segmentation"""
    parser = argparse.ArgumentParser(description="SAM2 + Grounding-DINO Segmentation for Estimator CV Demo")
    parser.add_argument("--input", "-i", required=True, help="Input image file or directory")
    parser.add_argument("--output", "-o", required=True, help="Output directory for results")
    parser.add_argument("--sam2-checkpoint", help="Path to SAM2 checkpoint file")
    parser.add_argument("--grounding-dino-config", help="Path to Grounding-DINO config file")
    
    args = parser.parse_args()
    
    try:
        segmenter = SAM2GroundingSegmenter(
            sam2_checkpoint=args.sam2_checkpoint,
            grounding_dino_config=args.grounding_dino_config
        )
        
        if os.path.isfile(args.input):
            # Process single image
            result = segmenter.interactive_segmentation(args.input, args.output)
            print(f"\n✅ Segmentation complete!")
            print(f"📊 Generated {len(result)} segmentations")
            for prompt, seg_result in result.items():
                print(f"  • {prompt}: {seg_result['visualization_path']}")
            
        elif os.path.isdir(args.input):
            # Process directory
            results = segmenter.process_directory(args.input, args.output)
            print(f"\n✅ Segmentation complete!")
            print(f"📁 Processed {len(results)} images")
            
        else:
            print(f"Error: {args.input} is not a valid file or directory")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Segmentation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
