import os
import random
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import json
import warnings
warnings.filterwarnings("ignore")
import gradio as gr

# --- 1. Parameters and Paths ---
original_dir = os.path.expanduser("~/Desktop/Plantas/PlantVillage")
splits = ['train', 'val', 'test']
split_ratios = (0.8, 0.1, 0.1)

IMG_SIZE = 300
BATCH_SIZE = 32
EPOCHS_HEAD = 10
EPOCHS_FINETUNE = 10

# --- 2. Create train/val/test folders and copy images ---
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

# --- 3. Create data generators with augmentation ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    width_shift_range=0.3,
    height_shift_range=0.3,
    zoom_range=0.3,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
    vertical_flip=True,
    channel_shift_range=50,
    shear_range=0.2
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
class_names = list(train_generator.class_indices.keys())

# --- 4. Enhanced Model Architecture with Attention ---
base_model = EfficientNetV2B0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
attention = layers.Dense(256, activation='sigmoid')(x) 
x = layers.multiply([x, attention])
predictions = layers.Dense(NUM_CLASSES, activation='softmax')(x)
model = models.Model(inputs=base_model.input, outputs=predictions)

# --- 5. Class Weights ---
class_weights = compute_class_weight('balanced', classes=np.unique(train_generator.classes), y=train_generator.classes)
class_weights = dict(enumerate(class_weights))

# --- 6. Callbacks and Learning Rate Schedule ---
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
initial_learning_rate = 1e-4
decay_steps = EPOCHS_HEAD * len(train_generator)

callbacks = [
    EarlyStopping(patience=15, restore_best_weights=True, monitor='val_accuracy'),
    ReduceLROnPlateau(factor=0.1, patience=5, monitor='val_loss'),
    TensorBoard(log_dir=log_dir, histogram_freq=1),
    tf.keras.callbacks.ModelCheckpoint('models/best_model.h5', save_best_only=True)
]

# --- 7. Training Head with Cosine Decay ---
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate, decay_steps)
optimizer = optimizers.Adam(lr_schedule)

model.compile(optimizer=optimizer, 
              loss='categorical_crossentropy', 
              metrics=['accuracy', 
                      tf.keras.metrics.Precision(name='precision'),
                      tf.keras.metrics.Recall(name='recall')])

model.fit(train_generator, 
          epochs=EPOCHS_HEAD, 
          validation_data=val_generator, 
          callbacks=callbacks, 
          class_weight=class_weights, 
          verbose=2)

# --- 8. Fine-tuning ---
model = tf.keras.models.load_model('models/best_model.V2')
for layer in model.layers[-20:]:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True

model.compile(optimizer=optimizers.Adam(1e-5), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

model.fit(train_generator, 
          validation_data=val_generator, 
          epochs=EPOCHS_FINETUNE, 
          callbacks=callbacks, 
          class_weight=class_weights, 
          verbose=2)

# --- 9. Enhanced Evaluation ---
def tta_predict(model, generator, steps=5):
    preds = []
    for _ in range(steps):
        preds.append(model.predict(generator, verbose=0))
    return np.mean(preds, axis=0)

y_pred = tta_predict(model, test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

loss, acc, precision, recall = model.evaluate(test_generator)
print(f"\n🎯 Test accuracy: {acc:.4f}")
print(f"🎯 Test precision: {precision:.4f}")
print(f"🎯 Test recall: {recall:.4f}")

print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred_classes, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
os.makedirs('reports', exist_ok=True)
plt.savefig('reports/confusion_matrix.png')
plt.close()

# ROC Curve
y_test = label_binarize(y_true, classes=np.arange(NUM_CLASSES))
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(NUM_CLASSES):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
plt.figure()
for i in range(NUM_CLASSES):
    plt.plot(fpr[i], tpr[i], label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig('reports/roc_curve.png')
plt.close()

# --- 10. Save final model and classes ---
model.save('models/efficientnetb3_plantvillage_finetuned.V2')
with open('models/class_names.json', 'w') as f:
    json.dump(train_generator.class_indices, f)

# --- 11. Model Quantization ---
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()
with open('models/efficientnetV2_plantvillage_quant.tflite', 'wb') as f:
    f.write(tflite_quant_model)

# --- 12. Enhanced Gradio Interface ---
treatment_info = {
    "healthy": {
        "diagnosis": "Healthy plant",
        "treatment": "No treatment needed. Maintain proper watering and sunlight.",
        "prevention": "Continue current care regimen with regular monitoring."
    },
    "powdery_mildew": {
        "diagnosis": "Powdery Mildew infection",
        "treatment": "Apply fungicides containing sulfur or potassium bicarbonate. Remove severely infected leaves.",
        "prevention": "Improve air circulation, avoid overhead watering, and maintain proper plant spacing."
    },
    # Add more treatments as needed
}

def predict_image(img):
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    class_idx = np.argmax(prediction)
    
    # Get top 3 predictions
    top_classes = np.argsort(prediction[0])[-3:][::-1]
    top_predictions = {class_names[i]: float(prediction[0][i]) for i in top_classes}
    
    # Get treatment info
    current_class = class_names[class_idx]
    treatment = treatment_info.get(current_class, {
        "diagnosis": current_class,
        "treatment": "Consult with a plant pathologist for specific treatment options.",
        "prevention": "Maintain good plant hygiene and monitor regularly."
    })
    
    return {
        "predictions": top_predictions,
        "diagnosis": treatment["diagnosis"],
        "treatment": treatment["treatment"],
        "prevention": treatment["prevention"],
        "confidence": f"{prediction[0][class_idx]*100:.1f}%"
    }

demo = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="Upload Plant Leaf Image"),
    outputs=[
        gr.Label(num_top_classes=3, label="Top Predictions"),
        gr.Textbox(label="Diagnosis"),
        gr.Textbox(label="Recommended Treatment"),
        gr.Textbox(label="Prevention Tips"),
        gr.Textbox(label="Confidence Level")
    ],
    title="🌱 Plant Disease Classifier",
    description="Upload an image of a plant leaf to diagnose potential diseases and get treatment recommendations",
    examples=[["examples/healthy.jpg"], ["examples/diseased.jpg"]],
    theme="soft",
    allow_flagging="never"
)

demo.launch()