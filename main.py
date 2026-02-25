import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ---------- Custom Colors ----------
st.markdown("""
    <style>
    .stApp {
        background-color: #ECE7D1;
    }
    h1 {
        color: #8A7650;
        text-align: center;
    }
    .stButton>button {
        background-color: #8E977D;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 18px;
    }
    .stFileUploader {
        background-color: #DBCEA5;
        padding: 10px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ---------- Load Model ----------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cat_dog_model.h5")

model = load_model()

st.title("🐶🐱 Cat vs Dog Classifier")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg","webp","avif","jfif"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    if st.button("Predict"):
        img = image.resize((224,224))
        img_array = np.array(img)/255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)

        if prediction[0][0] > 0.5:
            st.success("🐶 DOG")
        else:
            st.success("🐱 CAT")