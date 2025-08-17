#!/usr/bin/env python3
"""
Main Demo Script for Estimator CV Demo
Orchestrates the complete pipeline: PDF → Layout Detection → Segmentation → Classification
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import logging
import time
from datetime import datetime

# Import our modules
from pdf_processor import PDFProcessor
from layout_detector import LayoutDetector
from sam2_grounding import SAM2GroundingSegmenter
from cubicasa_demo import CubiCasaClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EstimatorCVDemo:
    """Main orchestrator for the Estimator CV Demo pipeline"""
    
    def __init__(self, output_base_dir: str = "demo_outputs"):
        """
        Initialize the demo pipeline
        
        Args:
            output_base_dir: Base directory for all outputs
        """
        self.output_base_dir = output_base_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(output_base_dir, f"session_{self.timestamp}")
        
        # Initialize components
        self.pdf_processor = None
        self.layout_detector = None
        self.segmenter = None
        self.classifier = None
        
        # Create output directories
        self._create_output_dirs()
        
        logger.info(f"Demo session initialized: {self.session_dir}")
    
    def _create_output_dirs(self):
        """Create output directory structure"""
        dirs = [
            self.session_dir,
            os.path.join(self.session_dir, "01_pdf_processed"),
            os.path.join(self.session_dir, "02_layout_detected"),
            os.path.join(self.session_dir, "03_segmented"),
            os.path.join(self.session_dir, "04_classified"),
            os.path.join(self.session_dir, "final_results")
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
    
    def run_pdf_processing(self, pdf_path: str, dpi: int = 300, deskew: bool = True) -> List[str]:
        """
        Step 1: PDF processing and rasterization
        
        Args:
            pdf_path: Path to input PDF
            dpi: DPI for rasterization
            deskew: Whether to apply deskewing
            
        Returns:
            List of paths to processed images
        """
        logger.info("=" * 60)
        logger.info("STEP 1: PDF Processing & Rasterization")
        logger.info("=" * 60)
        
        try:
            self.pdf_processor = PDFProcessor(dpi=dpi)
            processed_images = self.pdf_processor.process_pdf(
                pdf_path, 
                os.path.join(self.session_dir, "01_pdf_processed"),
                deskew=deskew
            )
            
            logger.info(f"✅ PDF processing complete: {len(processed_images)} images generated")
            return processed_images
            
        except Exception as e:
            logger.error(f"❌ PDF processing failed: {e}")
            raise
    
    def run_layout_detection(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Step 2: Layout detection using LayoutParser
        
        Args:
            image_paths: List of paths to processed images
            
        Returns:
            List of layout detection results
        """
        logger.info("=" * 60)
        logger.info("STEP 2: Layout Detection with LayoutParser")
        logger.info("=" * 60)
        
        try:
            self.layout_detector = LayoutDetector(model_name="PubLayNet")
            layout_results = []
            
            for image_path in image_paths:
                logger.info(f"Detecting layout in: {os.path.basename(image_path)}")
                result = self.layout_detector.process_image(
                    image_path, 
                    os.path.join(self.session_dir, "02_layout_detected")
                )
                layout_results.append(result)
                
                # Show quick stats
                stats = result['statistics']
                logger.info(f"  📊 Detected {stats['total_regions']} regions")
                logger.info(f"  🏠 Figures: {stats['figure_regions']}, Tables: {stats['table_regions']}, Text: {stats['text_regions']}")
            
            logger.info(f"✅ Layout detection complete: {len(layout_results)} images processed")
            return layout_results
            
        except Exception as e:
            logger.error(f"❌ Layout detection failed: {e}")
            raise
    
    def run_segmentation(self, image_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Step 3: Interactive segmentation with SAM2 + Grounding-DINO
        
        Args:
            image_paths: List of paths to processed images
            
        Returns:
            List of segmentation results
        """
        logger.info("=" * 60)
        logger.info("STEP 3: Interactive Segmentation (SAM2 + Grounding-DINO)")
        logger.info("=" * 60)
        
        try:
            # Note: This requires model checkpoints to be downloaded
            logger.info("Initializing SAM2 + Grounding-DINO models...")
            logger.info("Note: Model checkpoints need to be downloaded separately")
            
            # For demo purposes, we'll create a mock segmenter
            # In practice, you would initialize the real models here
            segmentation_results = []
            
            for image_path in image_paths:
                logger.info(f"Processing segmentation for: {os.path.basename(image_path)}")
                
                # Create mock segmentation result for demo
                mock_result = {
                    'image_path': image_path,
                    'segmentations': {
                        'room': {'visualization_path': f"{image_path}_room_seg.png"},
                        'door': {'visualization_path': f"{image_path}_door_seg.png"},
                        'click_center': {'visualization_path': f"{image_path}_click_seg.png"}
                    }
                }
                
                segmentation_results.append(mock_result)
                logger.info(f"  📊 Generated 3 segmentation examples")
            
            logger.info(f"✅ Segmentation complete: {len(segmentation_results)} images processed")
            logger.info("Note: This is a mock demonstration. Install model checkpoints for real segmentation.")
            return segmentation_results
            
        except Exception as e:
            logger.error(f"❌ Segmentation failed: {e}")
            logger.info("Continuing with demo...")
            return []
    
    def run_classification(self, image_paths: List[str], use_real_api: bool = False) -> List[Dict[str, Any]]:
        """
        Step 4: Floor plan classification with CubiCasa5k
        
        Args:
            image_paths: List of paths to processed images
            use_real_api: Whether to use real Roboflow API or demo mode
            
        Returns:
            List of classification results
        """
        logger.info("=" * 60)
        logger.info("STEP 4: Floor Plan Classification (CubiCasa5k)")
        logger.info("=" * 60)
        
        try:
            self.classifier = CubiCasaClassifier(api_key='ZdkZuHA7GF5NdOfCwQo8')
            classification_results = []
            
            for image_path in image_paths:
                logger.info(f"Classifying: {os.path.basename(image_path)}")
                
                if use_real_api:
                    # Use real API if available
                    try:
                        result = self.classifier.process_image(
                            image_path, 
                            os.path.join(self.session_dir, "04_classified"),
                            confidence_threshold=0.25
                        )
                        classification_results.append(result)
                        logger.info(f"  📊 Detected {result['classification']['total_detections']} elements")
                    except Exception as e:
                        logger.warning(f"  ⚠️  Real API failed, falling back to demo: {e}")
                        result = self.classifier.demo_without_api(
                            image_path, 
                            os.path.join(self.session_dir, "04_classified")
                        )
                        classification_results.append(result)
                else:
                    # Use demo mode
                    result = self.classifier.demo_without_api(
                        image_path, 
                        os.path.join(self.session_dir, "04_classified")
                    )
                    classification_results.append(result)
                
                logger.info(f"  🏠 Room count: {result['analysis']['room_count']}")
                logger.info(f"  🚪 Door count: {result['analysis']['door_count']}")
                logger.info(f"  🪟 Window count: {result['analysis']['window_count']}")
            
            logger.info(f"✅ Classification complete: {len(classification_results)} images processed")
            return classification_results
            
        except Exception as e:
            logger.error(f"❌ Classification failed: {e}")
            raise
    
    def generate_summary_report(self, all_results: Dict[str, Any]) -> str:
        """
        Generate a comprehensive summary report
        
        Args:
            all_results: Dictionary containing all pipeline results
            
        Returns:
            Path to the generated report
        """
        logger.info("=" * 60)
        logger.info("GENERATING SUMMARY REPORT")
        logger.info("=" * 60)
        
        report_path = os.path.join(self.session_dir, "final_results", "demo_summary_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("ESTIMATOR CV DEMO - SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Session: {self.timestamp}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # PDF Processing Summary
            f.write("1. PDF PROCESSING\n")
            f.write("-" * 20 + "\n")
            if 'pdf_results' in all_results:
                f.write(f"Input PDF: {all_results['pdf_results']['input_pdf']}\n")
                f.write(f"Images generated: {len(all_results['pdf_results']['processed_images'])}\n")
                f.write(f"DPI: {all_results['pdf_results']['dpi']}\n")
                f.write(f"Deskewing: {'Applied' if all_results['pdf_results']['deskew'] else 'Skipped'}\n\n")
            
            # Layout Detection Summary
            f.write("2. LAYOUT DETECTION\n")
            f.write("-" * 20 + "\n")
            if 'layout_results' in all_results:
                total_regions = sum(r['statistics']['total_regions'] for r in all_results['layout_results'])
                f.write(f"Images processed: {len(all_results['layout_results'])}\n")
                f.write(f"Total regions detected: {total_regions}\n")
                f.write(f"Model used: PubLayNet\n\n")
            
            # Segmentation Summary
            f.write("3. INTERACTIVE SEGMENTATION\n")
            f.write("-" * 20 + "\n")
            if 'segmentation_results' in all_results:
                f.write(f"Images processed: {len(all_results['segmentation_results'])}\n")
                f.write("Methods: SAM2 (click-based) + Grounding-DINO (text-based)\n")
                f.write("Note: Demo mode - install model checkpoints for real segmentation\n\n")
            
            # Classification Summary
            f.write("4. FLOOR PLAN CLASSIFICATION\n")
            f.write("-" * 20 + "\n")
            if 'classification_results' in all_results:
                total_elements = sum(r['analysis']['total_elements'] for r in all_results['classification_results'])
                total_rooms = sum(r['analysis']['room_count'] for r in all_results['classification_results'])
                total_doors = sum(r['analysis']['door_count'] for r in all_results['classification_results'])
                total_windows = sum(r['analysis']['window_count'] for r in all_results['classification_results'])
                
                f.write(f"Images processed: {len(all_results['classification_results'])}\n")
                f.write(f"Total architectural elements: {total_elements}\n")
                f.write(f"Rooms detected: {total_rooms}\n")
                f.write(f"Doors detected: {total_doors}\n")
                f.write(f"Windows detected: {total_windows}\n")
                f.write("Model: Roboflow CubiCasa5k\n\n")
            
            # File Locations
            f.write("5. OUTPUT FILES\n")
            f.write("-" * 20 + "\n")
            f.write(f"Session directory: {self.session_dir}\n")
            f.write("Subdirectories:\n")
            f.write("  • 01_pdf_processed/ - Rasterized images\n")
            f.write("  • 02_layout_detected/ - Layout detection results\n")
            f.write("  • 03_segmented/ - Segmentation masks\n")
            f.write("  • 04_classified/ - Classification results\n")
            f.write("  • final_results/ - Summary and reports\n\n")
            
            # Next Steps
            f.write("6. NEXT STEPS\n")
            f.write("-" * 20 + "\n")
            f.write("• Install SAM2 and Grounding-DINO model checkpoints for real segmentation\n")
            f.write("• Get Roboflow API key for real-time classification\n")
            f.write("• Integrate with your estimation workflow\n")
            f.write("• Customize prompts and parameters for your domain\n")
        
        logger.info(f"✅ Summary report generated: {report_path}")
        return report_path
    
    def run_complete_pipeline(self, pdf_path: str, dpi: int = 300, deskew: bool = True, 
                             use_real_api: bool = False) -> Dict[str, Any]:
        """
        Run the complete CV pipeline
        
        Args:
            pdf_path: Path to input PDF
            dpi: DPI for rasterization
            deskew: Whether to apply deskewing
            use_real_api: Whether to use real APIs or demo mode
            
        Returns:
            Dictionary with all pipeline results
        """
        start_time = time.time()
        
        logger.info("🚀 STARTING ESTIMATOR CV DEMO PIPELINE")
        logger.info(f"📁 Input: {pdf_path}")
        logger.info(f"📁 Output: {self.session_dir}")
        logger.info(f"⚙️  DPI: {dpi}, Deskew: {deskew}")
        logger.info(f"🌐 Real API: {use_real_api}")
        
        all_results = {}
        
        try:
            # Step 1: PDF Processing
            processed_images = self.run_pdf_processing(pdf_path, dpi, deskew)
            all_results['pdf_results'] = {
                'input_pdf': pdf_path,
                'processed_images': processed_images,
                'dpi': dpi,
                'deskew': deskew
            }
            
            # Step 2: Layout Detection
            layout_results = self.run_layout_detection(processed_images)
            all_results['layout_results'] = layout_results
            
            # Step 3: Segmentation
            segmentation_results = self.run_segmentation(processed_images)
            all_results['segmentation_results'] = segmentation_results
            
            # Step 4: Classification
            classification_results = self.run_classification(processed_images, use_real_api)
            all_results['classification_results'] = classification_results
            
            # Generate Summary Report
            report_path = self.generate_summary_report(all_results)
            all_results['summary_report'] = report_path
            
            # Calculate total time
            total_time = time.time() - start_time
            
            logger.info("=" * 60)
            logger.info("🎉 PIPELINE COMPLETE!")
            logger.info("=" * 60)
            logger.info(f"⏱️  Total time: {total_time:.1f} seconds")
            logger.info(f"📁 Results saved to: {self.session_dir}")
            logger.info(f"📋 Summary report: {report_path}")
            
            return all_results
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            raise

def main():
    """Main command line interface"""
    parser = argparse.ArgumentParser(description="Estimator CV Demo - Complete Pipeline")
    parser.add_argument("--pdf", "-p", required=True, help="Input PDF file path")
    parser.add_argument("--output", "-o", default="demo_outputs", help="Output base directory")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for rasterization (default: 300)")
    parser.add_argument("--no-deskew", action="store_true", help="Skip deskewing step")
    parser.add_argument("--real-api", action="store_true", help="Use real APIs instead of demo mode")
    
    args = parser.parse_args()
    
    try:
        # Initialize demo
        demo = EstimatorCVDemo(output_base_dir=args.output)
        
        # Run complete pipeline
        results = demo.run_complete_pipeline(
            pdf_path=args.pdf,
            dpi=args.dpi,
            deskew=not args.no_deskew,
            use_real_api=args.real_api
        )
        
        print(f"\n🎯 Demo completed successfully!")
        print(f"📁 Check results in: {demo.session_dir}")
        print(f"📋 Summary report: {results['summary_report']}")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
