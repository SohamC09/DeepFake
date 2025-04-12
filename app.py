import streamlit as st
import tensorflow as tf
import numpy as np
import gdown
import os
import cv2
from PIL import Image
import tempfile

# Google Drive model download setup
MODEL_PATH = "model.h5"
FILE_ID = "1CzG9lrRAHHZ0wNjr2lhLovKhSh7mJ3Pp"  # Replace with your real file ID

@st.cache_resource
def download_and_load_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = download_and_load_model()

st.title("Image & Video Classification App 📷🎥")

uploaded_file = st.file_uploader("Upload an image or a video file", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

def preprocess_image(image):
    image = image.resize((224, 224))  # or the input size your model expects
    img_array = np.array(image) / 255.0
    if len(img_array.shape) == 2:  # grayscale
        img_array = np.stack((img_array,) * 3, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

if uploaded_file is not None:
    file_type = uploaded_file.type

    if "image" in file_type:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        input_data = preprocess_image(image)
        prediction = model.predict(input_data)[0]
        pred_class = np.argmax(prediction)
        confidence = np.max(prediction)
        st.success(f"Predicted Class: {pred_class} (Confidence: {confidence:.2f})")

    elif "video" in file_type:
        # Save video temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())

        cap = cv2.VideoCapture(tfile.name)
        success, frame = cap.read()
        if success:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            st.image(image, caption="Extracted Frame from Video", use_column_width=True)
            input_data = preprocess_image(image)
            prediction = model.predict(input_data)[0]
            pred_class = np.argmax(prediction)
            confidence = np.max(prediction)
            st.success(f"Predicted Class: {pred_class} (Confidence: {confidence:.2f})")
        else:
            st.error("Could not read frame from video.")

        cap.release()
