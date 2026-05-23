import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# Load model
model = tf.keras.models.load_model(
    r"Model_2026_05_16_11_48_09_.h5"
)

st.title("MNIST Digit Prediction App")

uploaded_file = st.file_uploader(
    "Upload Digit Image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("L")

    st.image(image, width=150)

    # Resize
    image = image.resize((28, 28))

    # Convert to array
    img = np.array(image)

    # Invert image
    img = 255 - img

    # Normalize
    img = img / 255.0

    # Reshape
    img = img.reshape(1, 28, 28, 1)

    # Prediction
    pred = model.predict(img)

    digit = np.argmax(pred)

    st.success(f"Predicted Digit: {digit}")