from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image
import io
import base64
import os
import tensorflow as tf

app = Flask(__name__)

# Load model
MODEL_PATH = "model/fire_model.h5"
model = None

def load_model():
    global model
    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully!")
    else:
        print("⚠️ Model not found. Train the model first using model/train_model.py")

def preprocess_image(image):
    image = image.resize((64, 64))
    image = np.array(image) / 255.0
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
        if model is None:
            return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500

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
        prediction = model.predict(processed)[0][0]

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
    load_model()
    app.run(debug=True)