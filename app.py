import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable MKL optimizations for CPU

import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from utils.gradcam import get_cpu_safe_gradcam  # CPU-safe Grad-CAM

tf.config.run_functions_eagerly(True)  # Safe for CPU gradients


# PAGE CONFIG

st.set_page_config(page_title="Skin AI", layout="wide")


# CUSTOM CSS

st.markdown("""
<style>
.main { background-color: #f8f9fa; }
.title { text-align: center; font-size: 40px; font-weight: bold; color: #2c3e50; }
.subtitle { text-align: center; font-size: 18px; color: gray; }
.result-box { padding: 20px; border-radius: 10px; background-color: #fff; box-shadow: 0px 4px 10px rgba(0,0,0,0.1); }
</style>
""", unsafe_allow_html=True)


# HEADER

st.markdown("<div class='title'>🧠 Skin Disease Detection AI</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload an image and let AI analyze your skin condition</div>", unsafe_allow_html=True)
st.write("")


# SIDEBAR

st.sidebar.title("⚙️ Options")
st.sidebar.write("Upload an image and click predict")


# LOAD MODEL

model = tf.keras.models.load_model("models/skin_disease_model.h5")
class_names = ['Melanoma','Nevus','BCC','AK','BKL','DF','Vascular']


# DISEASE INFO

disease_info = {
    "Melanoma": "A serious type of skin cancer originating in melanocytes. "
                "It often appears as a new mole or changes in an existing mole. "
                "Early detection is critical for treatment. "
                "Symptoms include asymmetrical moles, irregular borders, and color variations. "
                "Avoid excessive sun exposure and perform regular skin checks.",
    "Nevus": "Commonly known as a mole, usually benign. "
             "Can be flat or raised and varies in color. "
             "Most nevi are harmless, but changes in size, shape, or color should be checked. "
             "Regular monitoring is recommended, especially for people with fair skin.",
    "BCC": "Basal Cell Carcinoma is the most common skin cancer. "
           "It develops in the basal cells of the skin, usually due to sun exposure. "
           "It grows slowly and rarely spreads, but early treatment is advised. "
           "Look for pearly or waxy bumps and visible blood vessels on the skin.",
    "AK": "Actinic Keratosis is a precancerous patch caused by sun damage. "
          "Appears as rough, scaly spots on sun-exposed areas. "
          "Can develop into squamous cell carcinoma if untreated. "
          "Regular dermatology check-ups are recommended.",
    "BKL": "Benign Keratosis-Like lesions are non-cancerous growths. "
           "Appear as rough, brown or black scaly patches. "
           "Mostly cosmetic concern; usually painless. "
           "Monitor for any sudden changes in appearance.",
    "DF": "Dermatofibroma is a harmless fibrous skin nodule. "
          "Usually appears as firm, small bumps, often on limbs. "
          "No treatment needed unless for cosmetic reasons. "
          "They remain stable in size and color.",
    "Vascular": "Vascular lesions involve abnormal blood vessels in the skin. "
                "Includes birthmarks, hemangiomas, and spider veins. "
                "Can be raised or flat, red or purple. "
                "Most are harmless; treatment is optional for cosmetic concerns."
}


# FILE UPLOAD

uploaded_file = st.file_uploader("📤 Upload Skin Image", type=["jpg","png"])

if uploaded_file:
    col1, col2 = st.columns(2)

    # Show uploaded image
    with col1:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

    # Prediction & Grad-CAM
    with col2:
        if st.button("🔍 Predict"):
            with st.spinner("Analyzing image..."):
                # Preprocess image
                img = image.resize((224,224))
                img_array = np.expand_dims(np.array(img)/255.0, axis=0)

               
                # PREDICTION
               
                prediction = model.predict(img_array)
                pred_class = class_names[np.argmax(prediction)]
                confidence = np.max(prediction)

            # Show result
            st.markdown("<div class='result-box'>", unsafe_allow_html=True)
            st.success(f"🩺 Prediction: {pred_class}")
            st.info(f"📊 Confidence: {confidence*100:.2f}%")
            st.markdown("</div>", unsafe_allow_html=True)

            # Show disease info
            if pred_class in disease_info:
                st.write("")
                st.subheader("ℹ️ About this Disease")
                st.write(disease_info[pred_class])

           
            # CPU-SAFE GRAD-CAM
           
            try:
                gradcam_img = get_cpu_safe_gradcam(model, img_array)
                st.write("")
                st.subheader("🔥 Model Focus Area (Grad-CAM)")
                st.image(gradcam_img, use_container_width=True)
            except Exception as e:
                st.warning("⚠️ Grad-CAM failed to run.")
                print("Grad-CAM error:", e)