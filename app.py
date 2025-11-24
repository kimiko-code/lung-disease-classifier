import os
import gdown
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from PIL import Image
import requests  # <-- NEW: for downloading from Google Drive

# Initialize Flask app
app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Google Drive model config
MODEL_URL = "https://drive.google.com/uc?export=download&id=1lSU56GjhA-s1ypk_74MqDOW28iMDU8yw"
MODEL_PATH = "lung_model.keras"

# Corrected class names (from notebook class_indices)
CLASS_NAMES = ['NORMAL', 'PNEUMONIA', 'TUBERCULOSIS']

# Global model variable (lazy-loaded)
model = None


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def preprocess_image(image_path):
    """Preprocess the image to match model requirements"""
    img = Image.open(image_path)

    # Convert grayscale to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')

    img = img.resize((224, 224))  # Input size used in training
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    return img_array


def download_model_if_needed():
    """Download the model file from Google Drive if not present locally."""
    if os.path.exists(MODEL_PATH):
        return

    print("Model file not found locally. Downloading from Google Drive...")
    try:
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        print("Model downloaded successfully.")
    except Exception as e:
        print(f"Error downloading model: {e}")
        raise RuntimeError(f"Failed to download model: {e}")


def get_model():
    """Lazy-load the model (download + load only once)."""
    global model
    if model is None:
        download_model_if_needed()
        print("Loading model from disk...")
        # If you get warnings about custom objects / compile, you can add: compile=False
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully.")
    return model


@app.route('/')
def home():
    """Render the HTML interface"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only PNG, JPG, JPEG allowed.'}), 400

    try:
        # Save the file temporarily
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess and predict
        processed_image = preprocess_image(filepath)

        # Ensure model is loaded (and downloaded if needed)
        loaded_model = get_model()
        predictions = loaded_model.predict(processed_image)

        predicted_index = np.argmax(predictions)
        predicted_class = CLASS_NAMES[predicted_index].strip()  # Strip leading/trailing spaces
        confidence = float(np.max(predictions))

        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence,
            'class_probabilities': {
                CLASS_NAMES[i].strip(): float(prob) for i, prob in enumerate(predictions[0])
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        # Clean up - remove the uploaded file
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)


if __name__ == '__main__':
    # Create uploads folder if it doesn't exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Run the app (debug=True for development only)
    app.run(host='0.0.0.0', port=5000, debug=False)
