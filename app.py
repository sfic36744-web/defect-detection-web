from flask import Flask, request, render_template, jsonify, send_from_directory
import os
import uuid
from werkzeug.utils import secure_filename
from optimized_defect_detection import intelligent_defect_detection
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'defect_reports'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/detect', methods=['POST'])
def api_detect_defects():
    """API endpoint for defect detection"""
    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file uploaded'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        if file and allowed_file(file.filename):
            # Generate unique filename
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            
            print(f"🔄 Processing image: {filename}")
            
            # Process the image
            result = intelligent_defect_detection(filepath, output_dir=app.config['RESULT_FOLDER'])
            
            if result is None:
                # Clean up uploaded file
                if os.path.exists(filepath):
                    os.remove(filepath)
                return jsonify({
                    'success': False,
                    'error': 'Error processing image - models may not be loaded correctly'
                }), 500
            
            # Prepare response data
            response_data = {
                'success': True,
                'data': {
                    'id': str(uuid.uuid4()),
                    'timestamp': datetime.now().isoformat(),
                    'filename': filename,
                    'uploaded_image': unique_filename,
                    'results': {
                        'final_type': result['final_type'],
                        'final_confidence': float(result['final_confidence']),
                        'fusion_reason': result['fusion_reason'],
                        'defect_percentage': float(result['defect_percentage']),
                        'defect_count': result['defect_count'],
                        'defects_info': result['defects_info'],
                        'report_path': os.path.basename(result['report_path']) if result['report_path'] else None,
                        'result_image_path': os.path.basename(result['result_image_path']) if result['result_image_path'] else None
                    }
                }
            }
            
            print(f"✅ Analysis completed: {result['final_type']}")
            
            return jsonify(response_data)
        
        else:
            return jsonify({
                'success': False,
                'error': 'File type not allowed. Please upload an image file (PNG, JPG, JPEG, BMP)'
            }), 400
            
    except Exception as e:
        print(f"❌ Error in API: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/detect', methods=['GET', 'POST'])
def detect_defects():
    """Web interface for defect detection"""
    if request.method == 'POST':
        try:
            if 'file' not in request.files:
                return render_template('result.html', 
                                    error='No file uploaded')
            
            file = request.files['file']
            
            if file.filename == '':
                return render_template('result.html', 
                                    error='No file selected')
            
            if file and allowed_file(file.filename):
                # Generate unique filename
                filename = secure_filename(file.filename)
                unique_filename = f"{uuid.uuid4()}_{filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(filepath)
                
                print(f"🔄 Processing image: {filename}")
                
                # Process the image
                result = intelligent_defect_detection(filepath, output_dir=app.config['RESULT_FOLDER'])
                
                if result is None:
                    # Clean up uploaded file
                    if os.path.exists(filepath):
                        os.remove(filepath)
                    return render_template('result.html', 
                                        error='Error processing image. Please check if model files are available.')
                
                # Convert paths to relative for web display
                if result['report_path']:
                    result['report_path'] = os.path.basename(result['report_path'])
                if result['result_image_path']:
                    result['result_image_path'] = os.path.basename(result['result_image_path'])
                
                print(f"✅ Analysis completed: {result['final_type']}")
                
                return render_template('result.html', 
                                    result=result,
                                    uploaded_image=unique_filename,
                                    success=True)
            
            else:
                return render_template('result.html', 
                                    error='File type not allowed. Please upload an image file.')
        
        except Exception as e:
            print(f"❌ Error in web interface: {str(e)}")
            # Clean up uploaded file if exists
            if 'filepath' in locals() and os.path.exists(filepath):
                os.remove(filepath)
            return render_template('result.html', 
                                error=f'Error processing image: {str(e)}')
    
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<filename>')
def result_file(filename):
    """Serve result files"""
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'Defect Detection API'
    })

@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'success': False,
        'error': 'File too large. Maximum size is 16MB.'
    }), 413

if __name__ == '__main__':
    print("🚀 Starting Defect Detection Flask Application...")
    print("📁 Upload folder:", app.config['UPLOAD_FOLDER'])
    print("📁 Result folder:", app.config['RESULT_FOLDER'])
    app.run(debug=False, host='0.0.0.0', port=5000)