import os
import gdown

MODEL_PATH = "brain_tumor_model.h5"

if not os.path.exists(MODEL_PATH):
    file_id = "1DYgOG7tPFulJfuW-OodqHAMRIpMI94z3"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, MODEL_PATH, quiet=False)
import os
import numpy as np
from PIL import Image, ImageOps
import streamlit as st
import tensorflow as tf

# Set page config
st.set_page_config(page_title="Brain Tumor MRI Classifier", layout="centered")

st.title("Brain Tumor MRI Classifier")
st.write("Upload a brain MRI scan to detect potential tumors.")

# Using forward slashes avoids all Windows unicode escape errors
MODEL_PATH = "C:/Users/babuv/Downloads/brain_app/brain_tumor_model.h5"

# 1. Load trained model with error handling
@st.cache_resource
def load_mri_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)

model = load_mri_model()

if model is None:
    st.error(f"Model file not found at: '{MODEL_PATH}'. Ensure you saved your model as 'brain_tumor_model.h5' in your project folder.")
    st.stop()

# Define class labels
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# 2. Image Uploader Widget
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI Scan", use_container_width=True)
    # Preprocessing function for standard input (224x224)
    def preprocess_image(img):
        target_size = (224, 224)
        img = ImageOps.fit(img, target_size, Image.Resampling.LANCZOS)
        
        img_array = np.asarray(img, dtype=np.float32)
        img_array = img_array / 255.0  # Scale pixel values between 0 and 1
        
        # Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    if st.button("Classify Scan"):
        with st.spinner("Analyzing image..."):
            processed_img = preprocess_image(image)
            
            # 3. Make Prediction
            predictions = model.predict(processed_img)
            predicted_index = np.argmax(predictions[0])
            predicted_class = CLASS_NAMES[predicted_index]
            confidence = float(np.max(predictions[0]) * 100)

            st.success("Analysis Complete!")
            st.write(f"### Result: **{predicted_class}**")
            st.write(f"Confidence: **{confidence:.2f}%**")
