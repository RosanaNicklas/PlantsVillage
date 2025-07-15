import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import os

# Rutas del dataset
train_dir = 'PlantVillage/train'
val_dir = 'PlantVillage/val'
test_dir = 'PlantVillage/test'

# Generador con aumento de datos
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

# Cargar datos
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_gen = train_datagen.flow_from_directory(train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')
val_gen = val_test_datagen.flow_from_directory(val_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')
test_gen = val_test_datagen.flow_from_directory(test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)


def plot_samples(generator, n=6):
    x, y = next(generator)
    plt.figure(figsize=(10, 5))
    for i in range(n):
        plt.subplot(2, 3, i+1)
        plt.imshow(x[i])
        plt.title(f"Class {np.argmax(y[i])}")
        plt.axis('off')
    plt.show()

plot_samples(train_gen)  # Mostrar algunas imágenes del conjunto de entrenamiento [[3]]

# Usar MobileNetV2 preentrenado
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Congelar capas

x = GlobalAveragePooling2D()(base_model.output)
x = Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
output = Dense(train_gen.num_classes, activation='softmax')(x)

model = Model(base_model.input, output)

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(
    train_gen,
    epochs=10,
    validation_data=val_gen
)


model.save('plantvillage_mobilenet.h5')  # Guardar el modelo entrenado [[7]]
print("Modelo guardado como plantvillage_mobilenet.h5")


# Cargar modelo guardado
loaded_model = load_model('plantvillage_mobilenet.h5')

# Probar una imagen de ejemplo
example_img_path = os.path.join(test_dir, os.listdir(test_dir)[0], os.listdir(os.path.join(test_dir, os.listdir(test_dir)[0]))[0])
img = load_img(example_img_path, target_size=IMG_SIZE)
img_array = img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

pred = loaded_model.predict(img_array)
predicted_class = np.argmax(pred, axis=1)

plt.imshow(plt.imread(example_img_path))
plt.title(f"Predicción: Clase {predicted_class[0]}")
plt.axis('off')
plt.show()

import pandas as pd

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.xlabel("Épocas")
plt.ylabel("Valor")
plt.title("Curvas de entrenamiento")
plt.show()