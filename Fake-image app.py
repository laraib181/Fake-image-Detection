import streamlit as st
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import tensorflow as tf
import os

# Load the pre-trained model
model = tf.keras.models.load_model('model.h5')

# ELA function
def convert_to_ela_image(image_path_or_img, quality=90):
    temp_file = 'temp_ela.jpg'
    if isinstance(image_path_or_img, str):
        original = Image.open(image_path_or_img).convert('RGB')
    else:
        original = image_path_or_img.convert('RGB')

    original.save(temp_file, 'JPEG', quality=quality)
    compressed = Image.open(temp_file)
    ela_image = ImageChops.difference(original, compressed)
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0 / max_diff if max_diff != 0 else 1
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    os.remove(temp_file)
    return ela_image

# Streamlit UI
st.title("Fake Image Detection Using ELA + Deep Learning")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Original Image', use_column_width=True)

    try:
        st.write("Converting to ELA image...")
        ela_img = convert_to_ela_image(image)
        st.image(ela_img, caption='ELA Image', use_column_width=True)

        ela_img = ela_img.resize((128, 128))
        img_array = tf.keras.preprocessing.image.img_to_array(ela_img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)

        st.write("Making prediction...")
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_index]

        class_labels = {0: 'Real (Authentic)', 1: 'Fake (Tampered)'}
        predicted_label = class_labels.get(predicted_class_index, 'Unknown')

        st.success(f"Prediction: {predicted_label}")
        st.info(f"Confidence: {confidence:.2f}")

    except Exception as e:
        st.error(f"Error: {e}")
