#!/usr/bin/env python3
"""
Quick Start Script for Estimator CV Demo
Easy way to get started with minimal setup
"""

import os
import sys
import argparse
from pathlib import Path

def main():
    """Quick start interface"""
    print("🚀 ESTIMATOR CV DEMO - QUICK START")
    print("=" * 50)
    
    # Check if we have a PDF to work with
    pdf_files = list(Path(".").glob("*.pdf"))
    
    if not pdf_files:
        print("❌ No PDF files found in current directory")
        print("\nTo get started:")
        print("1. Place a PDF file (architectural drawing/floor plan) in this directory")
        print("2. Run this script again")
        print("3. Or run: python demo.py --pdf your_file.pdf")
        return
    
    print(f"📄 Found {len(pdf_files)} PDF file(s):")
    for i, pdf_file in enumerate(pdf_files):
        print(f"  {i+1}. {pdf_file.name}")
    
    # Ask user which PDF to use
    if len(pdf_files) == 1:
        selected_pdf = pdf_files[0]
        print(f"\n✅ Using: {selected_pdf.name}")
    else:
        try:
            choice = input(f"\nSelect PDF (1-{len(pdf_files)}): ").strip()
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(pdf_files):
                selected_pdf = pdf_files[choice_idx]
                print(f"✅ Selected: {selected_pdf.name}")
            else:
                print("❌ Invalid choice")
                return
        except (ValueError, KeyboardInterrupt):
            print("\n❌ Invalid input")
            return
    
    # Check system requirements
    print("\n🔍 Checking system requirements...")
    try:
        from utils import check_system_requirements, print_system_info
        requirements = check_system_requirements()
        
        if not all(requirements.values()):
            print("\n⚠️  Some requirements are missing. Install with:")
            print("   pip install -r requirements.txt")
            print("\nContinue anyway? (y/n): ", end="")
            try:
                if input().lower() != 'y':
                    return
            except KeyboardInterrupt:
                return
    except ImportError:
        print("⚠️  Could not check requirements")
    
    # Run the demo
    print(f"\n🚀 Starting demo with {selected_pdf.name}...")
    print("This will:")
    print("  1. Convert PDF to high-resolution images")
    print("  2. Detect layout regions (text, figures, tables)")
    print("  3. Run interactive segmentation (demo mode)")
    print("  4. Classify architectural elements (demo mode)")
    
    try:
        # Import and run demo
        from demo import EstimatorCVDemo
        
        demo = EstimatorCVDemo(output_base_dir="quick_demo_outputs")
        results = demo.run_complete_pipeline(
            pdf_path=str(selected_pdf),
            dpi=300,
            deskew=True,
            use_real_api=False
        )
        
        print(f"\n🎉 Demo completed successfully!")
        print(f"📁 Results saved to: {demo.session_dir}")
        print(f"📋 Summary report: {results['summary_report']}")
        
        # Show next steps
        print("\n📚 NEXT STEPS:")
        print("• Check the output directory for results")
        print("• Install model checkpoints for real segmentation")
        print("• Get Roboflow API key for real-time classification")
        print("• Customize for your specific use case")
        
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("Make sure all dependencies are installed:")
        print("   pip install -r requirements.txt")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("Check the error message above for details")

if __name__ == "__main__":
    main()
