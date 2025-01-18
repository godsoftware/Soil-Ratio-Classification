# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 21:42:42 2024

@author: ozkal
"""
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import os
import cv2
import seaborn as sns
import tensorflow as tf
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# GPU Bellek Yönetimi
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Dinamik GPU Belleği Yönetimi
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
        
# Veri seti yolları
train_dir = "C:/capstoneproject/Capstonedata/train"
val_dir = "C:/capstoneproject/Capstonedata/validation"
test_dir = "C:/capstoneproject/Capstonedata/test"

# Görüntü boyutu ve parametreler
image_size = (224, 224)  # Görüntü boyutu
batch_size = 32  # Mini batch boyutu

# 1. Veri yükleme ve artırma
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    brightness_range=[0.8, 1.2],  # Parlaklık ayarlaması
    zoom_range=0.2,  # Zoom işlemi
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1.0/255)    # Normalizasyon
test_datagen = ImageDataGenerator(rescale=1.0/255)   # Normalizasyon


train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False  # Test seti sıralı olmalı
)

# 2. CNN Modeli Tanımlama
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(train_generator.class_indices), activation='softmax')  # Sınıf sayısı kadar çıktı
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 3. Model Eğitimi
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,  # Epoch sayısı
    steps_per_epoch=train_generator.samples // batch_size,
    validation_steps=val_generator.samples // batch_size
)

# 4. Test Setinde Değerlendirme
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# 5. Modeli Kaydetme (Belirli Klasöre ve Özelleştirilmiş Ad ile)
save_dir = "C:/capstoneproject/models"  # Modelin kaydedileceği klasör
if not os.path.exists(save_dir):
    os.makedirs(save_dir)  # Klasör yoksa oluştur

model_name = f"model_epoch{len(history.history['accuracy'])}_batch{batch_size}_cnn.keras"
save_path = os.path.join(save_dir, model_name)
model.save(save_path)
print(f"Model şu dizine kaydedildi: {save_path}")

# 6. Confusion Matrix için Test Sonuçlarının Tahmini
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)

# Gerçek sınıfları karşılaştırma
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Confusion Matrix
conf_matrix = confusion_matrix(true_classes, predicted_classes)

# Confusion Matrix Görselleştirme
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Tahmin Edilen Sınıf")
plt.ylabel("Gerçek Sınıf")
plt.title("Confusion Matrix")
plt.show()

# Eğitim ve Kayıp Sonuçlarının Görselleştirilmesi
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
plt.plot(history.history['loss'], label='Train Loss', color='green')
plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
plt.legend()
plt.title('Loss and Accuracy')
plt.show()
