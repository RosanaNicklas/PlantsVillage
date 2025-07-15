import tensorflow as tf
import numpy as np
import json
import gradio as gr
from PIL import Image

IMG_SIZE = 300
model = tf.keras.models.load_model("efficientnetb3_plantvillage_finetuned.h5")
with open("class_names.json") as f:
    class_indices = json.load(f)
class_names = list(class_indices.keys())

def predict_image(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)[0]
    return {class_names[i]: float(pred[i]) for i in range(len(class_names))}

gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=3),
    title="🌿 Plant Disease Classifier",
    description="Upload a plant leaf image to predict the disease."
).launch()
 
"""git init
git remote add origin https://huggingface.co/spaces/your-username/plant-disease
git add .
git commit -m "Initial commit"
git push -u origin main"""