import os
import random
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0  # Or change to B3 if needed
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import json

# --- 1. Parámetros y rutas ---
original_dir = os.path.expanduser("~/Desktop/Plantas/PlantVillage")
splits = ['train', 'val', 'test']
split_ratios = (0.8, 0.1, 0.1)

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_HEAD = 10
EPOCHS_FINETUNE = 10

# --- 2. Crear carpetas train/val/test y copiar imágenes ---
print("=== Separando dataset en train/val/test ===")

for split in splits:
    os.makedirs(os.path.join(original_dir, split), exist_ok=True)

for class_name in os.listdir(original_dir):
    class_path = os.path.join(original_dir, class_name)
    if not os.path.isdir(class_path) or class_name in splits:
        continue

    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(images)

    n_total = len(images)
    n_train = int(n_total * split_ratios[0])
    n_val = int(n_total * split_ratios[1])

    split_dict = {
        'train': images[:n_train],
        'val': images[n_train:n_train+n_val],
        'test': images[n_train+n_val:]
    }

    for split, split_images in split_dict.items():
        split_dir = os.path.join(original_dir, split, class_name)
        os.makedirs(split_dir, exist_ok=True)
        for img_name in split_images:
            src = os.path.join(class_path, img_name)
            dst = os.path.join(split_dir, img_name)
            if os.path.exists(src):
                shutil.copy2(src, dst)
            else:
                print(f"⚠️ Archivo no encontrado: {src}")

    print(f"✅ {class_name}: {n_total} imágenes divididas en train/val/test")

# --- 3. Crear generadores ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    width_shift_range=0.3,
    height_shift_range=0.3,
    zoom_range=0.3,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
    vertical_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(original_dir, 'train'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(original_dir, 'val'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = val_datagen.flow_from_directory(
    os.path.join(original_dir, 'test'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

NUM_CLASSES = len(train_generator.class_indices)
print(f"Clases: {train_generator.class_indices}")

# --- 4. Definir modelo EfficientNetB0 base + cabeza ---
base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = True

# Congelar algunas capas
for layer in base_model.layers[:100]:
    layer.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# --- 5. Class Weights ---
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights))

# --- 6. Callbacks ---
callbacks = [
    EarlyStopping(patience=15, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.1, patience=5)
]

# --- 7. Entrenamiento principal ---
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("=== Entrenando modelo completo ===")
history = model.fit(
    train_generator,
    epochs=EPOCHS_HEAD,
    validation_data=val_generator,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=2
)

model.save('efficientnetb0_plantvillage_base.h5')

# --- 8. Fine-tuning ---
print("=== Fine-tuning últimas capas ===")
model = tf.keras.models.load_model('efficientnetb0_plantvillage_base.h5')

for layer in model.layers[-20:]:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True

model.compile(optimizer=optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS_FINETUNE,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=2
)

# --- 9. Evaluar en test ---
print("=== Evaluando en test set ===")
loss, acc = model.evaluate(test_generator)
print(f"🎯 Test accuracy: {acc:.4f}")

# --- 10. Guardar modelo final + clases ---
model.save('efficientnetb0_plantvillage_finetuned.h5')

with open('class_names.json', 'w') as f:
    json.dump(train_generator.class_indices, f)

# --- 11. Convertir a TFLite ---
print("=== Convirtiendo a TFLite ===")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('efficientnetb0_plantvillage.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ Proceso completo, modelo guardado y convertido.")
