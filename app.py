from pathlib import Path

import cv2
import joblib
import numpy as np
import streamlit as st
from PIL import Image
from skimage.feature import hog


IMG_SIZE = (128, 128)
MODEL_PATH = Path("outputs/svm_crop_disease_model.joblib")

DISPLAY_NAMES = {
    "healthy": "Healthy",
    "scab": "Apple Scab",
    "black_rot": "Black Rot",
    "cedar_apple_rust": "Cedar Apple Rust",
}

DESCRIPTIONS = {
    "healthy": "The model classifies this leaf as healthy.",
    "scab": "The model detects visual patterns associated with apple scab.",
    "black_rot": "The model detects visual patterns associated with black rot.",
    "cedar_apple_rust": "The model detects visual patterns associated with cedar apple rust.",
}


def extract_features(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, IMG_SIZE)
    return hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
    )


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.set_page_config(
    page_title="Crop Disease SVM Demo",
    page_icon="🌿",
    layout="centered",
)

st.title("🌿 AI Crop Disease Classification")
st.subheader("Live Demo — SVM Model")

st.markdown(
    """
This demo classifies **apple leaf images** into four classes using a trained
**Support Vector Machine (SVM)** model.

**Classes:** Healthy · Apple Scab · Black Rot · Cedar Apple Rust
"""
)

if not MODEL_PATH.exists():
    st.error(
        "The trained model was not found. First run: "
        "`python train_svm.py --data data/apple --max-per-class 1000`"
    )
    st.stop()

model = load_model()

uploaded = st.file_uploader(
    "Upload an apple leaf image",
    type=["jpg", "jpeg", "png"],
)

if uploaded is None:
    st.info("Upload a leaf image to start the prediction demo.")
    st.markdown(
        """
### How the prediction works
1. The image is resized to **128 × 128**.
2. It is converted to grayscale.
3. **HOG** extracts numerical image features.
4. The trained **RBF SVM** receives those features.
5. The SVM returns the most likely disease/health class.
"""
    )
else:
    image_pil = Image.open(uploaded).convert("RGB")
    st.image(image_pil, caption="Input leaf image", use_container_width=True)

    image_rgb = np.array(image_pil)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    features = extract_features(image_bgr).reshape(1, -1)

    prediction = model.predict(features)[0]

    st.success(f"Prediction: {DISPLAY_NAMES.get(prediction, prediction)}")
    st.write(DESCRIPTIONS.get(prediction, "Prediction completed."))

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        classes = model.classes_
        order = np.argsort(probabilities)[::-1]

        st.markdown("### Confidence scores")
        for idx in order:
            label = DISPLAY_NAMES.get(classes[idx], classes[idx])
            st.progress(float(probabilities[idx]), text=f"{label}: {probabilities[idx]*100:.1f}%")

    st.markdown("### What the SVM did")
    st.write(
        "The uploaded image went through the same preprocessing and HOG feature "
        "extraction used during training. The trained SVM then evaluated those "
        "features against the decision boundaries it learned from the training data."
    )

st.divider()
st.caption("Demo project: AI-Based Crop Disease Classification using SVM")
