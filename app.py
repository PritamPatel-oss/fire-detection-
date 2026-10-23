from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
import io
import base64
import os

# Support lightweight TFLite runtime or full TensorFlow fallback
try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter

app = Flask(__name__)

# Model paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TFLITE_PATH = os.path.join(BASE_DIR, "model", "fire_model.tflite")
H5_PATH = os.path.join(BASE_DIR, "model", "fire_model.h5")

interpreter = None
input_details = None
output_details = None
h5_model = None

def load_model():
    global interpreter, input_details, output_details, h5_model
    if os.path.exists(TFLITE_PATH):
        interpreter = Interpreter(model_path=TFLITE_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print("Model loaded successfully (TensorFlow Lite - Low Memory)!")
    elif os.path.exists(H5_PATH):
        import tensorflow as tf
        h5_model = tf.keras.models.load_model(H5_PATH)
        print("Model loaded successfully (H5 fallback)!")
    else:
        print("Warning: Model file not found.")

# Load model on startup
load_model()

def preprocess_image(image):
    image = image.resize((64, 64))
    image = np.array(image, dtype=np.float32) / 255.0
    if image.shape[-1] == 4:  # Remove alpha channel if present
        image = image[:, :, :3]
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if interpreter is None and h5_model is None:
            return jsonify({'error': 'Model not loaded. Please train or provide model file.'}), 500

        data = request.get_json()
        image_data = data['image']

        # Decode base64 image
        if ',' in image_data:
            image_data = image_data.split(',')[1]

        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        # Preprocess
        processed = preprocess_image(image)

        # Predict
        if interpreter is not None:
            interpreter.set_tensor(input_details[0]['index'], processed)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
        else:
            prediction = h5_model.predict(processed)[0][0]

        fire_detected = prediction < 0.5
        confidence = float(prediction) if fire_detected else float(1 - prediction)
        confidence_pct = round(confidence * 100, 2)

        return jsonify({
            'fire_detected': bool(fire_detected),
            'confidence': confidence_pct,
            'raw_score': float(prediction)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
