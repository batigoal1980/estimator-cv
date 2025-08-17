# Quick Estimator Drawing CV Demo

A fast demonstration of computer vision capabilities for processing architectural drawings and floor plans through PDF analysis, layout detection, and interactive segmentation.

## 🚀 Quick Demo Recipe (10-15 minutes)

1. **PDF Processing**: Rasterize PDF pages (300-600 DPI) and deskew
2. **Layout Detection**: Run LayoutParser to show detected regions (figures/tables/text) as boxes
3. **Interactive Segmentation**: Add SAM2 + Grounding-DINO for click or text-prompt segmentation
4. **Optional**: Try Roboflow's CubiCasa5k model for floor-plan-aware classification

## 🛠️ Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install system dependencies (macOS)
brew install tesseract poppler

# Install system dependencies (Ubuntu)
sudo apt-get install tesseract-ocr poppler-utils
```

## 📁 Project Structure

```
Estimator-CV/
├── requirements.txt          # Python dependencies
├── demo.py                  # Main demo script
├── pdf_processor.py         # PDF rasterization and deskewing
├── layout_detector.py       # LayoutParser integration
├── sam2_grounding.py       # SAM2 + Grounding-DINO integration
├── cubicasa_demo.py         # Roboflow CubiCasa5k integration
├── utils.py                 # Utility functions
├── app.py                   # Flask web application
├── start_web.py             # Web interface starter
├── templates/               # HTML templates
│   ├── index.html          # Main upload page
│   └── demo.html           # Demo progress page
├── uploads/                 # Uploaded PDF files
├── web_results/             # Web demo results
└── samples/                 # Sample PDFs and outputs
```

## 🎯 Usage

### Quick Start
```bash
python demo.py --pdf path/to/your/drawing.pdf
```

### 🌐 Web Interface (Recommended)
```bash
# Start the web interface
python start_web.py

# Or run directly
python app.py
```
Then open your browser to `http://localhost:5000`

### Step-by-Step Demo
```bash
# 1. PDF Processing
python pdf_processor.py --input drawing.pdf --output processed_images/

# 2. Layout Detection
python layout_detector.py --input processed_images/ --output layouts/

# 3. Interactive Segmentation
python sam2_grounding.py --input processed_images/ --output segments/

# 4. Floor Plan Classification (Optional)
python cubicasa_demo.py --input processed_images/ --output classifications/
```

## 🔧 Features

- **High-Resolution PDF Processing**: 300-600 DPI rasterization with automatic deskewing
- **Layout Detection**: Automatic detection of figures, tables, and text regions
- **Interactive Segmentation**: Click-to-segment or text-prompt segmentation
- **Floor Plan Awareness**: Specialized models for architectural drawings
- **Real-time Visualization**: Instant feedback and results display

## 📊 Output Examples

- **Layout Boxes**: Bounding boxes around detected regions
- **Segmentation Masks**: Precise masks for selected elements
- **Classification Results**: Room types and architectural elements
- **Interactive Interface**: Click and prompt-based segmentation

## 🎨 Customization

- Adjust DPI settings for different quality requirements
- Modify layout detection parameters for specific document types
- Customize segmentation prompts for domain-specific terminology
- Integrate with your existing estimation workflow

## 📝 Notes

- First run will download model weights automatically
- GPU acceleration recommended for SAM2 and Grounding-DINO
- Test with sample architectural drawings for best results
- Adjust memory settings for large PDF files
