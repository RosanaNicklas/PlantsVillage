# app.py (Flask Backend + Streamlit Frontend Hybrid)
import numpy as np
from PIL import Image
import tensorflow as tf
from flask import Flask, request, jsonify
import streamlit as st
import os

# ======================
# 🏗️ CONFIGURATION
# ======================
MODEL_PATH = "plantvillage_mobilenet.h5"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# ======================
# 🌱 PLANTVILLE CLASS MAPPING (38 classes)
# ======================
CLASS_NAMES = {
    0: "Tomato - Bacterial spot",
    1: "Tomato - Early blight",
    2: "Tomato - Late blight", 
    3: "Tomato - Leaf Mold",
    4: "Tomato - Septoria leaf spot",
    5: "Tomato - Spider mites",
    6: "Tomato - Target Spot",
    7: "Tomato - Yellow Leaf Curl Virus",
    8: "Tomato - Mosaic virus",
    9: "Tomato - Healthy",
    10: "Potato - Early blight",
    11: "Potato - Late blight",
    12: "Potato - Healthy",
    13: "Pepper - Bacterial spot",
    14: "Pepper - Healthy",
    15: "Corn - Gray leaf spot",
    16: "Corn - Common rust",
    17: "Corn - Northern Leaf Blight", 
    18: "Corn - Healthy",
    19: "Strawberry - Leaf scorch",
    20: "Strawberry - Healthy",
    21: "Apple - Scab",
    22: "Apple - Black rot",
    23: "Apple - Cedar rust",
    24: "Apple - Healthy",
    25: "Cherry - Powdery mildew",
    26: "Cherry - Healthy",
    27: "Peach - Bacterial spot",
    28: "Peach - Healthy",
    29: "Grape - Black rot",
    30: "Grape - Black Measles",
    31: "Grape - Leaf blight",
    32: "Grape - Healthy",
    33: "Blueberry - Healthy",
    34: "Orange - Huanglongbing"
}

# ======================
# 🤖 MODEL LOADING
# ======================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    model.compile(optimizer='adam', 
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    return model

model = load_model()

# ======================
# 🖼️ IMAGE PROCESSING
# ======================
def preprocess_image(image):
    img = Image.open(image).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# ======================
# 🚀 FLASK API
# ======================
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def api_predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty file"}), 400
    
    try:
        img_array = preprocess_image(file)
        preds = model.predict(img_array)
        
        return jsonify({
            "class_index": int(np.argmax(preds)),
            "class_name": CLASS_NAMES[int(np.argmax(preds))],
            "confidence": float(np.max(preds)) * 100,
            "all_predictions": preds[0].tolist()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ======================
# 📱 STREAMLIT FRONTEND
# ======================
def streamlit_app():
    st.title("🌱 Plant Disease Classifier")
    st.markdown("Upload a leaf image for disease diagnosis")
    
    uploaded_file = st.file_uploader("Choose an image...", type=ALLOWED_EXTENSIONS)
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", width=300)
            
        with col2:
            with st.spinner('Analyzing...'):
                img_array = preprocess_image(uploaded_file)
                preds = model.predict(img_array)
                class_idx = np.argmax(preds)
                
                st.success(f"""
                **Diagnosis:** {CLASS_NAMES[class_idx]}  
                **Confidence:** {np.max(preds)*100:.2f}%
                """)
                
                # Show probability distribution
                st.bar_chart({
                    "Probability": preds[0]
                })

# ======================
# 🏃 RUN SYSTEM
# ======================
if __name__ == "__main__":
    # Run both Flask and Streamlit
    import threading
    
    # Start Flask in background thread
    threading.Thread(
        target=app.run,
        kwargs={'host': '0.0.0.0', 'port': 5000}
    ).start()
    
    # Run Streamlit
    streamlit_app()