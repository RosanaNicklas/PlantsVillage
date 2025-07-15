import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

IMG_SIZE = 300

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("plantvillage_mobilenet.h5")

@st.cache_resource
def load_class_names():
    with open("class_names.json") as f:
        class_indices = json.load(f)
    return list(class_indices.keys())

model = load_model()
class_names = load_class_names()

st.title("🌿 Plant Disease Classifier")
st.write("Upload a plant leaf image to detect its disease class.")

file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    img_resized = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)
    prediction = model.predict(img_array)[0]

    st.subheader("Prediction:")
    for i, prob in sorted(enumerate(prediction), key=lambda x: -x[1])[:3]:
        st.write(f"**{class_names[i]}**: {prob:.2%}")
