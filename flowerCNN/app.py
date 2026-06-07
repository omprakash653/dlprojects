import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# ── Config ──────────────────────────────────────────────────────────────────
IMG_HEIGHT = 180
IMG_WIDTH  = 180
CLASS_NAMES = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]

EMOJI = {
    "daisy":      "🌼",
    "dandelion":  "🌾",
    "roses":      "🌹",
    "sunflowers": "🌻",
    "tulips":     "🌷",
}

COLOR = {
    "daisy":      "#F5C842",
    "dandelion":  "#A8C256",
    "roses":      "#E8425A",
    "sunflowers": "#F5A623",
    "tulips":     "#C471ED",
}

# ── Page setup ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Flower Classifier",
    page_icon="🌸",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Main background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

/* Hero title */
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 3rem;
    color: #fff;
    text-align: center;
    margin-bottom: 0.2rem;
    letter-spacing: -0.5px;
}
.hero-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    color: rgba(255,255,255,0.55);
    text-align: center;
    margin-bottom: 2.5rem;
    font-weight: 300;
    letter-spacing: 0.5px;
}

/* Upload zone */
.upload-box {
    border: 2px dashed rgba(255,255,255,0.2);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(10px);
    transition: border-color 0.3s;
}

/* Result card */
.result-card {
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-top: 1.5rem;
    text-align: center;
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.12);
    animation: fadeUp 0.5s ease;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}

.result-emoji {
    font-size: 4rem;
    display: block;
    margin-bottom: 0.5rem;
}
.result-label {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    font-weight: 400;
    color: #fff;
    text-transform: capitalize;
    margin-bottom: 0.3rem;
}
.result-conf {
    font-size: 1rem;
    color: rgba(255,255,255,0.6);
    font-weight: 300;
}

/* Bar chart labels */
.bar-label {
    font-size: 0.8rem;
    color: rgba(255,255,255,0.7);
    text-transform: capitalize;
}

/* Divider */
.divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.08);
    margin: 2rem 0;
}

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.stFileUploader label { color: rgba(255,255,255,0.7) !important; }
</style>
""", unsafe_allow_html=True)


# ── Load model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("flower_classifier.keras", compile=False, safe_mode=False)
        return model, None
    except Exception as e:
        return None, str(e)


# ── Predict ───────────────────────────────────────────────────────────────────
def predict(model, img: Image.Image):
    img_resized = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_arr     = tf.keras.utils.img_to_array(img_resized)
    img_arr     = tf.expand_dims(img_arr, axis=0)          # add batch dim
    predictions = model.predict(img_arr, verbose=0)
    scores      = tf.nn.softmax(predictions[0]).numpy()
    return scores


# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🌸 Flower Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Powered by TensorFlow · CNN trained on 5 flower species</div>', unsafe_allow_html=True)

model, err = load_model()

if err:
    st.error(f"⚠️ Could not load **flower_classifier.keras**\n\n`{err}`\n\nMake sure the model file is in the same folder as this script.")
    st.stop()

uploaded = st.file_uploader(
    "Drop a flower image here",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed"
)

if uploaded:
    img = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.image(img, caption="Uploaded image", use_column_width=True)

    scores = predict(model, img)
    top_idx   = int(np.argmax(scores))
    top_label = CLASS_NAMES[top_idx]
    top_score = float(scores[top_idx])
    top_color = COLOR[top_label]
    top_emoji = EMOJI[top_label]

    with col2:
        st.markdown(f"""
        <div class="result-card" style="background: linear-gradient(135deg, {top_color}22, {top_color}11);">
            <span class="result-emoji">{top_emoji}</span>
            <div class="result-label">{top_label}</div>
            <div class="result-conf">{top_score*100:.1f}% confidence</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown("**All predictions**")

        # Sort by score descending
        sorted_idx = np.argsort(scores)[::-1]
        for i in sorted_idx:
            label = CLASS_NAMES[i]
            score = float(scores[i])
            bar_color = COLOR[label]
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.markdown(f'<span class="bar-label">{EMOJI[label]} {label}</span>', unsafe_allow_html=True)
                st.progress(score)
            with col_b:
                st.markdown(f'<span style="color:rgba(255,255,255,0.6);font-size:0.85rem;">{score*100:.1f}%</span>', unsafe_allow_html=True)

else:
    st.markdown("""
    <div class="upload-box">
        <div style="font-size:2.5rem; margin-bottom:0.5rem;">🌼 🌹 🌻</div>
        <div style="color:rgba(255,255,255,0.5); font-size:0.95rem;">
            Upload a JPG or PNG of a flower<br>
            <span style="font-size:0.8rem; opacity:0.6;">daisy · dandelion · rose · sunflower · tulip</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <br>
    <div style="text-align:center; color:rgba(255,255,255,0.25); font-size:0.78rem; letter-spacing:1px;">
        RUN WITH &nbsp;·&nbsp; <code style="background:rgba(255,255,255,0.08); padding:2px 8px; border-radius:4px;">streamlit run flower_classifier_app.py</code>
    </div>
    """, unsafe_allow_html=True)