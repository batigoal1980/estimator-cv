# 🚀 Quick Estimator Drawing CV Demo Guide (10-15 minutes)

This guide will walk you through the complete CV pipeline for processing architectural drawings and floor plans, from PDF to interactive segmentation, in just 10-15 minutes.

## 🎯 What You'll Accomplish

1. **PDF Processing**: Convert PDF to high-resolution images with automatic deskewing
2. **Layout Detection**: Use LayoutParser to detect regions (figures/tables/text) as boxes
3. **Interactive Segmentation**: Add SAM2 + Grounding-DINO for click/text-prompt segmentation
4. **Floor Plan Classification**: Try Roboflow's CubiCasa5k model for architectural elements

## ⚡ Quick Start (5 minutes)

### 1. Setup
```bash
# Clone or download the project
cd Estimator-CV

# Install dependencies
pip install -r requirements.txt

# Run setup (optional but recommended)
python setup.py
```

### 2. Run the Complete Demo
```bash
# Place a PDF file in the directory, then run:
python quick_start.py

# Or run the full pipeline directly:
python demo.py --pdf your_drawing.pdf
```

## 🔧 Step-by-Step Demo (10-15 minutes)

### Step 1: PDF Processing & Rasterization (2-3 minutes)
```bash
python pdf_processor.py --input your_drawing.pdf --output processed_images/ --dpi 300
```
**What happens:**
- Converts PDF to high-resolution images (300-600 DPI)
- Applies automatic deskewing using Hough Line Transform
- Creates clean, aligned images for processing

**Output:** High-quality rasterized images ready for analysis

### Step 2: Layout Detection with LayoutParser (3-4 minutes)
```bash
python layout_detector.py --input processed_images/ --output layouts/
```
**What happens:**
- Detects document regions using PubLayNet model
- Identifies figures, tables, text, and other elements
- Creates bounding boxes around detected regions
- Extracts text content using OCR

**Output:** Visualized layout with colored bounding boxes and text previews

### Step 3: Interactive Segmentation (3-4 minutes)
```bash
python sam2_grounding.py --input processed_images/ --output segments/
```
**What happens:**
- **SAM2**: Click-to-segment specific areas (rooms, doors, windows)
- **Grounding-DINO**: Text-prompt segmentation ("segment the kitchen", "find all doors")
- Creates precise masks for selected elements

**Output:** Segmentation masks and overlays for architectural elements

### Step 4: Floor Plan Classification (2-3 minutes)
```bash
python cubicasa_demo.py --input processed_images/ --output classified/ --demo
```
**What happens:**
- Uses Roboflow's CubiCasa5k model for floor-plan-aware classification
- Identifies room types, fixtures, and architectural elements
- Provides confidence scores and bounding boxes
- Analyzes floor plan characteristics (openness ratio, element counts)

**Output:** Classified architectural elements with analysis

## 🎨 Demo Variations

### Quick Demo Mode
```bash
# Run everything in demo mode (no API keys needed)
python demo.py --pdf your_drawing.pdf --output demo_outputs/
```

### Real-Time Mode
```bash
# Use real APIs and models
export ROBOFLOW_API_KEY="your_key_here"
python demo.py --pdf your_drawing.pdf --real-api
```

### Individual Component Testing
```bash
# Test just the PDF processor
python pdf_processor.py --input test.pdf --output test_images/ --dpi 400

# Test just layout detection
python layout_detector.py --input test_images/ --output test_layouts/

# Test just segmentation
python sam2_grounding.py --input test_images/ --output test_segments/
```

## 📊 Expected Results

### Layout Detection
- **Text regions**: Blue boxes with extracted text
- **Figures**: Purple boxes around drawings/plans
- **Tables**: Orange boxes around data tables
- **Instant "it works" moment**: Clear visualization of document structure

### Segmentation
- **Click-based**: Click anywhere to segment that area
- **Text-based**: "segment the living room" → precise room mask
- **Refinement**: Add more clicks to improve segmentation quality

### Classification
- **Room detection**: Kitchen, bedroom, bathroom, etc.
- **Element counting**: Doors, windows, walls, fixtures
- **Floor plan analysis**: Area calculations, openness ratios

## 🚨 Troubleshooting

### Common Issues
1. **Missing dependencies**: Run `pip install -r requirements.txt`
2. **System packages**: Install tesseract and poppler
3. **Model checkpoints**: Download SAM2 and Grounding-DINO models
4. **API keys**: Set environment variables for real-time processing

### Performance Tips
- Use GPU acceleration for SAM2 and Grounding-DINO
- Adjust DPI based on your needs (300-600 DPI)
- Process images in batches for large documents

## 🔮 Next Steps

### Production Use
1. **Install model checkpoints** for real segmentation
2. **Get Roboflow API key** for live classification
3. **Customize prompts** for your domain
4. **Integrate with estimation workflow**

### Customization
- Modify confidence thresholds
- Add custom architectural element types
- Integrate with your existing tools
- Scale to batch processing

## 📁 Project Structure
```
Estimator-CV/
├── demo.py              # Main orchestrator
├── quick_start.py       # Easy entry point
├── pdf_processor.py     # PDF → Images
├── layout_detector.py   # LayoutParser integration
├── sam2_grounding.py   # SAM2 + Grounding-DINO
├── cubicasa_demo.py     # Roboflow integration
├── utils.py             # Helper functions
├── setup.py             # Installation helper
├── requirements.txt     # Dependencies
└── config.yaml          # Configuration
```

## 🎉 Success Metrics

**10-15 minute demo should show:**
- ✅ PDF converted to high-res images
- ✅ Layout regions detected and visualized
- ✅ Interactive segmentation working
- ✅ Floor plan elements classified
- ✅ Clear understanding of the pipeline

**Ready for production when:**
- Models are downloaded and configured
- APIs are set up and tested
- Workflow is integrated with estimation tools
- Performance meets your requirements

---

**Start your demo now with:** `python quick_start.py`

This will give you the complete "it works" experience in under 15 minutes! 🚀
