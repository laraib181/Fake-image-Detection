from flask import Flask, request, render_template, jsonify
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import tensorflow as tf
import os

# Load the pre-trained model
model = tf.keras.models.load_model('model.h5')

app = Flask(__name__)

# ELA function (copied from your notebook)
def convert_to_ela_image(image_path_or_img, quality=90):
    temp_file = 'temp_ela.jpg'
    if isinstance(image_path_or_img, str):
        original = Image.open(image_path_or_img).convert('RGB')
    else: # Assume it's a PIL Image object
         original = image_path_or_img.convert('RGB')

    original.save(temp_file, 'JPEG', quality=quality)
    compressed = Image.open(temp_file)
    ela_image = ImageChops.difference(original, compressed)
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0 / max_diff if max_diff != 0 else 1
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    os.remove(temp_file) # Clean up temp file
    return ela_image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        try:
            # Read the image file
            img = Image.open(file.stream)

            # Convert to ELA image
            ela_img = convert_to_ela_image(img)

            # Resize and preprocess for the model
            ela_img = ela_img.resize((128, 128))
            img_array = tf.keras.preprocessing.image.img_to_array(ela_img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0
            img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)

            # Make prediction
            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class_index]

            # Map index to label (based on your generator's class_indices)
            # You might need to get class_indices from your generator or define them
            class_labels = {0: 'Real (Authentic)', 1: 'Fake (Tampered)'}
            predicted_label = class_labels.get(predicted_class_index, 'Unknown')

            return jsonify({
                'prediction': predicted_label,
                'confidence': float(confidence)
            })

        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Use this for development within Colab (requires ngrok or similar for external access)
    # !pip install flask flask-ngrok
    # from flask_ngrok import run_with_ngrok
    # run_with_ngrok(app)
    # app.run()

    # Use this for local development outside Colab
    app.run(debug=True)