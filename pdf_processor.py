#!/usr/bin/env python3
"""
PDF Processor for Estimator CV Demo
Handles high-resolution PDF rasterization and automatic deskewing
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple
import numpy as np
import cv2
from pdf2image import convert_from_path
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessor:
    """High-resolution PDF processor with deskewing capabilities"""
    
    def __init__(self, dpi: int = 300):
        self.dpi = dpi
        self.supported_formats = ['.pdf']
    
    def rasterize_pdf(self, pdf_path: str, output_dir: str) -> List[str]:
        """
        Convert PDF to high-resolution images
        
        Args:
            pdf_path: Path to input PDF
            output_dir: Directory to save rasterized images
            
        Returns:
            List of paths to generated images
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Converting PDF to images at {self.dpi} DPI...")
        
        try:
            # Convert PDF to images
            images = convert_from_path(
                pdf_path, 
                dpi=self.dpi,
                fmt='PNG',
                thread_count=4
            )
            
            image_paths = []
            for i, image in enumerate(images):
                # Check dimensions
                width, height = image.size
                logger.info(f"Page {i+1} original dimensions: {width}x{height}")
                
                # Rotate 90 degrees clockwise to fix the counter-clockwise rotation issue
                # PIL rotate: positive = counter-clockwise, negative = clockwise
                image = image.rotate(-90, expand=True)
                logger.info(f"Rotated page {i+1} by 90 degrees clockwise to correct orientation")
                
                # Save image
                image_path = os.path.join(output_dir, f"page_{i+1:03d}.png")
                image.save(image_path, 'PNG')
                image_paths.append(image_path)
                
                logger.info(f"Saved page {i+1} to {image_path}")
            
            return image_paths
            
        except Exception as e:
            logger.error(f"Error converting PDF: {e}")
            raise
    
    def deskew_image(self, image_path: str) -> str:
        """
        Automatically deskew an image using Hough Line Transform
        
        Args:
            image_path: Path to input image
            
        Returns:
            Path to deskewed image
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Find lines using Hough Transform
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is not None:
            # Calculate skew angle
            angles = []
            for line in lines[:10]:  # Use first 10 lines
                rho, theta = line[0]  # Extract rho and theta from the line array
                angle = theta * 180 / np.pi
                if angle < 45:
                    angles.append(angle)
                elif angle > 135:
                    angles.append(angle - 180)
                else:
                    angles.append(90 - angle)
            
            if angles:
                # Calculate median skew angle
                skew_angle = np.median(angles)
                logger.info(f"Detected skew angle: {skew_angle:.2f} degrees")
                
                # Apply rotation correction
                if abs(skew_angle) > 0.5:  # Only correct if skew > 0.5 degrees
                    height, width = image.shape[:2]
                    center = (width // 2, height // 2)
                    
                    # Create rotation matrix
                    rotation_matrix = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
                    
                    # Calculate new image dimensions
                    cos_val = abs(rotation_matrix[0, 0])
                    sin_val = abs(rotation_matrix[0, 1])
                    new_width = int((height * sin_val) + (width * cos_val))
                    new_height = int((height * cos_val) + (width * sin_val))
                    
                    # Adjust rotation matrix for new dimensions
                    rotation_matrix[0, 2] += (new_width / 2) - center[0]
                    rotation_matrix[1, 2] += (new_height / 2) - center[1]
                    
                    # Apply rotation
                    deskewed = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))
                    
                    # Save deskewed image
                    deskewed_path = image_path.replace('.png', '_deskewed.png')
                    cv2.imwrite(deskewed_path, deskewed)
                    
                    logger.info(f"Saved deskewed image to {deskewed_path}")
                    return deskewed_path
        
        # If no deskewing needed, return original path
        logger.info("No significant skew detected")
        return image_path
    
    def process_pdf(self, pdf_path: str, output_dir: str, deskew: bool = True) -> List[str]:
        """
        Complete PDF processing pipeline
        
        Args:
            pdf_path: Path to input PDF
            output_dir: Directory to save processed images
            deskew: Whether to apply deskewing
            
        Returns:
            List of paths to processed images
        """
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Step 1: Rasterize PDF
        image_paths = self.rasterize_pdf(pdf_path, output_dir)
        
        # Step 2: Deskew images if requested
        if deskew:
            logger.info("Applying deskewing...")
            processed_paths = []
            for image_path in image_paths:
                deskewed_path = self.deskew_image(image_path)
                processed_paths.append(deskewed_path)
        else:
            processed_paths = image_paths
        
        logger.info(f"Processing complete. {len(processed_paths)} images generated.")
        return processed_paths

def main():
    """Command line interface for PDF processing"""
    parser = argparse.ArgumentParser(description="PDF Processor for Estimator CV Demo")
    parser.add_argument("--input", "-i", required=True, help="Input PDF file path")
    parser.add_argument("--output", "-o", required=True, help="Output directory for images")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for rasterization (default: 300)")
    parser.add_argument("--no-deskew", action="store_true", help="Skip deskewing step")
    
    args = parser.parse_args()
    
    try:
        processor = PDFProcessor(dpi=args.dpi)
        processed_paths = processor.process_pdf(
            args.input, 
            args.output, 
            deskew=not args.no_deskew
        )
        
        print(f"\n✅ Processing complete!")
        print(f"📁 Output directory: {args.output}")
        print(f"🖼️  Generated {len(processed_paths)} images")
        print(f"🔧 DPI: {args.dpi}")
        print(f"📐 Deskewing: {'Enabled' if not args.no_deskew else 'Disabled'}")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
