from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)

# Cargar el modelo entrenado

model = tf.keras.models.load_model('model/plantvillage_mobilenet.h5')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Re-compila para evitar la advertencia
# Definir la ruta principal para mostrar el formulario web
@app.route('/')
def home():
    return render_template('index.html')

# Ruta para hacer predicciones
@app.route('/predict', methods=['POST'])
def predict():
    # Ruta donde se guardará la imagen
    upload_folder = 'static/uploads'
    
    # Crear la carpeta si no existe
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    # Obtener la imagen desde el formulario
    img_file = request.files['image']
    
    # Guardar temporalmente la imagen en la carpeta static/uploads
    img_path = os.path.join('static', 'uploads', img_file.filename)
    img_file.save(img_path)

    # Preprocesar la imagen para el modelo
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Hacer predicción
    prediction = model.predict(img_array)
    predicted_class = int(np.argmax(prediction, axis=1)[0])

    # Interpretar resultado (ajusta según tus clases)
    print("Índice de clase predicha:", predicted_class)
    class_names = ["Normal", "Diseased"]  # Asegúrate de que estas clases coincidan con tu modelo
    if predicted_class < len(class_names):
        result = class_names[predicted_class]
    else:
        result = "Clase desconocida"
        print("Número de clases en class_names:", len(class_names))
    result = class_names[predicted_class]

    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(debug=True)