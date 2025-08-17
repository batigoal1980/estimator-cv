#!/usr/bin/env python3
"""
Flask Web Application for Estimator CV Demo
Provides a web interface for the complete CV pipeline
"""

import os
import sys

# Set headless mode for OpenCV on server environments
if os.environ.get('RAILWAY_ENVIRONMENT'):
    os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
    os.environ['OPENCV_VIDEOIO_PRIORITY_GSTREAMER'] = '0'
import json
import logging
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
import threading
import time

# Import our demo modules
EstimatorCVDemo = None
error_msg = None
try:
    from demo import EstimatorCVDemo
except ImportError as import_error:
    error_msg = str(import_error)
    print(f"Error importing EstimatorCVDemo: {error_msg}")
    import traceback
    traceback.print_exc()
    # Create a dummy class to prevent NameError
    class EstimatorCVDemo:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(f"EstimatorCVDemo import failed: {error_msg}")

try:
    from utils import create_demo_sample_pdf, validate_pdf_file
except ImportError as e:
    print(f"Warning: Could not import utils: {e}")
    # Define dummy functions
    def create_demo_sample_pdf():
        return None
    def validate_pdf_file(file):
        return True

# Configure Flask app
app = Flask(__name__)
app.secret_key = 'estimator_cv_demo_secret_key_2024'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'web_results'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for demo status
demo_status = {
    'running': False,
    'progress': 0,
    'current_step': '',
    'results': None,
    'error': None
}

class WebDemoRunner:
    """Runs the demo in a separate thread and updates status"""
    
    def __init__(self):
        self.status = demo_status
        self.thread = None
    
    def run_demo(self, pdf_path, dpi, deskew, use_real_api):
        """Run the demo pipeline"""
        try:
            self.status['running'] = True
            self.status['progress'] = 0
            self.status['error'] = None
            self.status['current_step'] = 'Initializing...'
            
            # Create demo instance
            demo = EstimatorCVDemo(output_base_dir=app.config['RESULTS_FOLDER'])
            
            # Check if input is an image or PDF
            file_ext = os.path.splitext(pdf_path.lower())[1]
            
            if file_ext in ['.png', '.jpg', '.jpeg']:
                # For images, skip PDF processing and go straight to CubiCasa detection
                self.status['current_step'] = 'Preparing image...'
                self.status['progress'] = 25
                
                # Copy image to session directory
                import shutil
                os.makedirs(os.path.join(demo.session_dir, "01_pdf_processed"), exist_ok=True)
                image_name = os.path.basename(pdf_path)
                dest_path = os.path.join(demo.session_dir, "01_pdf_processed", image_name)
                shutil.copy2(pdf_path, dest_path)
                processed_images = [dest_path]
                
                # Skip layout detection and segmentation for direct image processing
                layout_results = []
                segmentation_results = []
                
                # Run CubiCasa classification directly
                self.status['current_step'] = 'Running CubiCasa floor plan detection...'
                self.status['progress'] = 50
                classification_results = demo.run_classification(processed_images, use_real_api=True)
            else:
                # Original PDF processing pipeline
                # Step 1: PDF Processing
                self.status['current_step'] = 'Processing PDF...'
                self.status['progress'] = 25
                processed_images = demo.run_pdf_processing(pdf_path, dpi, deskew)
                
                # Step 2: Layout Detection
                self.status['current_step'] = 'Detecting layout...'
                self.status['progress'] = 50
                layout_results = demo.run_layout_detection(processed_images)
                
                # Step 3: Segmentation
                self.status['current_step'] = 'Running segmentation...'
                self.status['progress'] = 75
                segmentation_results = demo.run_segmentation(processed_images)
                
                # Step 4: Classification
                self.status['current_step'] = 'Classifying elements...'
                self.status['progress'] = 90
                classification_results = demo.run_classification(processed_images, use_real_api)
            
            # Generate summary report
            self.status['current_step'] = 'Generating report...'
            self.status['progress'] = 95
            report_path = demo.generate_summary_report({
                'pdf_results': {
                    'input_pdf': pdf_path,
                    'processed_images': processed_images,
                    'dpi': dpi,
                    'deskew': deskew
                },
                'layout_results': layout_results,
                'segmentation_results': segmentation_results,
                'classification_results': classification_results
            })
            
            # Update status
            self.status['progress'] = 100
            self.status['current_step'] = 'Complete!'
            self.status['results'] = {
                'session_dir': demo.session_dir,
                'report_path': report_path,
                'processed_images': [os.path.basename(img) for img in processed_images],  # Just filenames
                'layout_results': len(layout_results),
                'segmentation_results': len(segmentation_results),
                'classification_results': len(classification_results)
            }
            
        except Exception as e:
            logger.error(f"Demo failed: {e}")
            self.status['error'] = str(e)
            self.status['current_step'] = 'Failed'
        finally:
            self.status['running'] = False
    
    def start_demo(self, pdf_path, dpi, deskew, use_real_api):
        """Start demo in background thread"""
        if self.thread and self.thread.is_alive():
            return False
        
        self.thread = threading.Thread(
            target=self.run_demo,
            args=(pdf_path, dpi, deskew, use_real_api)
        )
        self.thread.daemon = True
        self.thread.start()
        return True

# Global demo runner
demo_runner = WebDemoRunner()

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    if 'image_file' not in request.files:
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    file = request.files['image_file']
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('index'))
    
    allowed_extensions = {'.png', '.jpg', '.jpeg'}
    file_ext = os.path.splitext(file.filename.lower())[1]
    
    if file and file_ext in allowed_extensions:
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        flash(f'Image uploaded successfully: {filename}', 'success')
        return redirect(url_for('demo', filename=filename))
    else:
        flash('Please upload a PNG or JPG image file', 'error')
        return redirect(url_for('index'))

@app.route('/demo/<filename>')
def demo(filename):
    """Demo page for uploaded file"""
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        flash('File not found', 'error')
        return redirect(url_for('index'))
    
    return render_template('demo.html', filename=filename)

@app.route('/api/start_demo', methods=['POST'])
def start_demo():
    """Start the demo pipeline"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        use_real_api = data.get('use_real_api', True)  # Always use real API for images
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        
        # Start demo (dpi and deskew not needed for images)
        success = demo_runner.start_demo(filepath, 300, True, use_real_api)
        if success:
            return jsonify({'message': 'Demo started successfully'})
        else:
            return jsonify({'error': 'Demo already running'}), 400
            
    except Exception as e:
        logger.error(f"Error starting demo: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/demo_status')
def get_demo_status():
    """Get current demo status"""
    return jsonify(demo_status)

@app.route('/api/results')
def get_results():
    """Get demo results"""
    if demo_status['results']:
        return jsonify(demo_status['results'])
    else:
        return jsonify({'error': 'No results available'}), 404

@app.route('/download/<path:filename>')
def download_file(filename):
    """Download result files"""
    try:
        filepath = os.path.join(app.config['RESULTS_FOLDER'], filename)
        if os.path.exists(filepath):
            return send_file(filepath, as_attachment=True)
        else:
            flash('File not found', 'error')
            return redirect(url_for('index'))
    except Exception as e:
        flash(f'Error downloading file: {e}', 'error')
        return redirect(url_for('index'))

@app.route('/api/create_sample_pdf')
def create_sample_pdf():
    """Create a sample PDF for testing"""
    try:
        sample_pdf = create_demo_sample_pdf()
        if sample_pdf:
            # Copy to uploads folder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_filename = f"{timestamp}_sample_demo.pdf"
            new_filepath = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
            shutil.copy2(sample_pdf, new_filepath)
            
            return jsonify({
                'message': 'Sample PDF created successfully',
                'filename': new_filename
            })
        else:
            return jsonify({'error': 'Could not create sample PDF'}), 500
    except Exception as e:
        logger.error(f"Error creating sample PDF: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/uploaded_files')
def get_uploaded_files():
    """Get list of uploaded files"""
    try:
        files = []
        upload_dir = Path(app.config['UPLOAD_FOLDER'])
        for file_path in upload_dir.glob("*.pdf"):
            files.append({
                'filename': file_path.name,
                'size_mb': round(file_path.stat().st_size / (1024 * 1024), 2),
                'uploaded': datetime.fromtimestamp(file_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            })
        return jsonify({'files': files})
    except Exception as e:
        logger.error(f"Error getting uploaded files: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/system_info')
def get_system_info():
    """Get system information"""
    try:
        from utils import check_system_requirements
        requirements = check_system_requirements()
        
        system_info = {
            'python_version': sys.version,
            'platform': sys.platform,
            'requirements': requirements,
            'upload_folder': app.config['UPLOAD_FOLDER'],
            'results_folder': app.config['RESULTS_FOLDER']
        }
        return jsonify(system_info)
    except Exception as e:
        logger.error(f"Error getting system info: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/web_results/<path:filepath>')
def serve_result(filepath):
    """Serve result images from the web_results directory"""
    try:
        # Security check - prevent directory traversal
        safe_path = os.path.join('web_results', filepath)
        safe_path = os.path.abspath(safe_path)
        
        # Ensure the path is within RESULTS_FOLDER
        if not safe_path.startswith(os.path.abspath(app.config['RESULTS_FOLDER'])):
            return "Access denied", 403
            
        if os.path.exists(safe_path):
            directory = os.path.dirname(safe_path)
            filename = os.path.basename(safe_path)
            return send_from_directory(directory, filename)
        else:
            return "File not found", 404
    except Exception as e:
        logger.error(f"Error serving result file: {e}")
        return "Error serving file", 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    print("🚀 Starting Estimator CV Demo Web Interface...")
    print("📁 Upload folder:", app.config['UPLOAD_FOLDER'])
    print("📁 Results folder:", app.config['RESULTS_FOLDER'])
    
    # Use PORT from environment variable (for Railway) or default to 5002
    port = int(os.environ.get('PORT', 5002))
    print(f"🌐 Web interface will be available at: http://localhost:{port}")
    
    app.run(debug=True, host='0.0.0.0', port=port)
