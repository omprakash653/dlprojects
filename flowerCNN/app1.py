import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

IMG_HEIGHT  = 180
IMG_WIDTH   = 180
CLASS_NAMES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
EMOJI       = {"daisy":"🌼","dandelion":"🌾","roses":"🌹","sunflowers":"🌻","tulips":"🌷"}

st.set_page_config(page_title="Flower Classifier", page_icon="🌸")
st.title("🌸 Flower Classifier")
st.caption("CNN trained on 5 flower species · daisy · dandelion · rose · sunflower · tulip")

@st.cache_resource
def load_model():
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
        tf.keras.layers.RandomTranslation(0.1, 0.1),
    ])
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(180, 180, 3)),
        data_augmentation,
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(5, name='logits')
    ])
    model.build((None, 180, 180, 3))
    model.load_weights("flower_weights.weights.h5")  # ← fixed filename
    return model

def predict(model, img):
    img_arr = tf.keras.utils.img_to_array(img.resize((IMG_WIDTH, IMG_HEIGHT)))
    scores  = tf.nn.softmax(model.predict(tf.expand_dims(img_arr, 0), verbose=0)[0]).numpy()
    return scores

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load model: {e}")
    st.stop()

uploaded = st.file_uploader("Upload a flower image", type=["jpg", "jpeg", "png", "webp"])

if uploaded:
    img    = Image.open(uploaded).convert("RGB")
    scores = predict(model, img)
    top    = int(np.argmax(scores))

    col1, col2 = st.columns(2)
    with col1:
        st.image(img, use_column_width=True)
    with col2:
        st.subheader(f"{EMOJI[CLASS_NAMES[top]]} {CLASS_NAMES[top].capitalize()}")
        st.metric("Confidence", f"{scores[top]*100:.1f}%")
        st.divider()
        for i in np.argsort(scores)[::-1]:
            st.text(f"{EMOJI[CLASS_NAMES[i]]} {CLASS_NAMES[i]:<12} {scores[i]*100:.1f}%")
            st.progress(float(scores[i]))