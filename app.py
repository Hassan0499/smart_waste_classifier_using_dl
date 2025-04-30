from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import io

app = Flask(__name__, static_url_path='', static_folder='.')
CORS(app)

# Load model
model = tf.keras.models.load_model('waste_model.h5')

# Load class names from the saved Labels.txt
with open('Labels.txt', 'r') as f:
    class_names = f.read().splitlines()

@app.route('/')
def home():
    return send_from_directory('./templates', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    img = Image.open(file).convert('RGB')
    img = img.resize((300, 300))  # Resize to match model input
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
