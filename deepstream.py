import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import os

# Configuración de la página
st.set_page_config(
    page_title="Diagnóstico de Enfermedades en Plantas",
    page_icon="🌱",
    layout="wide"
)

# Título con estilo
st.markdown("""
    <h1 style='text-align: center; color: #2e8b57;'>
    🌿 Diagnóstico de Enfermedades en Plantas
    </h1>
    """, unsafe_allow_html=True)

st.write("""
    *Sube una imagen de una hoja para identificar posibles enfermedades usando inteligencia artificial.*
    """)

# --- Carga del Modelo ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('plantvillage_mobilenet.h5')

model = load_model()

# --- Mapeo Completo de Clases PlantVillage ---
CLASS_NAMES = {
    0: "Tomate - Mancha bacteriana",
    1: "Tomate - Tizón temprano",
    2: "Tomate - Tizón tardío",
    3: "Tomate - Moho foliar",
    4: "Tomate - Septoriosis",
    5: "Tomate - Ácaros",
    6: "Tomate - Mancha objetivo",
    7: "Tomate - Virus del mosaico",
    8: "Tomate - Virus del enrollamiento amarillo",
    9: "Tomate - Sano",
    10: "Papa - Tizón temprano",
    11: "Papa - Tizón tardío",
    12: "Papa - Sana",
    13: "Pimiento - Mancha bacteriana",
    14: "Pimiento - Sano",
    15: "Maíz - Mancha gris",
    16: "Maíz - Roya común",
    17: "Maíz - Tizón norteño",
    18: "Maíz - Sano",
    19: "Fresa - Quemadura foliar",
    20: "Fresa - Sana",
    21: "Manzana - Sarna",
    22: "Manzana - Podredumbre negra",
    23: "Manzana - Roya",
    24: "Manzana - Sana",
    25: "Cereza - Oídio",
    26: "Cereza - Sana",
    27: "Melocotón - Mancha bacteriana",
    28: "Melocotón - Sano",
    29: "Uva - Podredumbre negra",
    30: "Uva - Enfermedad de Esca",
    31: "Uva - Tizón foliar",
    32: "Uva - Sana",
    33: "Arándano - Sano",
    34: "Naranja - Huanglongbing"
}

# --- Interfaz de Usuario ---
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Procesamiento de la imagen
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen analizada", width=300)
    
    # Redimensionar y normalizar
    img = image.resize((224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predicción
    with st.spinner('🔍 Analizando la hoja...'):
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions) * 100
    
    # Resultados
    st.success("✅ Análisis completado")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Diagnóstico:** {CLASS_NAMES[predicted_class]}")
        st.markdown(f"**Confianza:** {confidence:.2f}%")
    
    with col2:
        # Gráfico de probabilidades
        st.bar_chart({
            "Probabilidad": predictions[0]
        })

    # --- Recomendaciones por enfermedad ---
    st.subheader("📌 Recomendaciones")
    if "Sano" in CLASS_NAMES[predicted_class]:
        st.success("La planta parece saludable. ¡Sigue con los cuidados habituales!")
    else:
        st.warning("**Posible enfermedad detectada.** Consulta con un agrónomo para confirmar y tratamiento específico.")

# --- Footer ---
st.markdown("---")
st.caption("Modelo entrenado con el dataset PlantVillage | MobileNetV2")

# Para ejecutar: streamlit run app_streamlit.py