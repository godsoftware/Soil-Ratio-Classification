# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 02:52:50 2024

@author: ozkal
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report


# Veri seti yolları
train_dir = "C:/capstoneproject/Capstonedata/train"
val_dir = "C:/capstoneproject/Capstonedata/validation"
test_dir = "C:/capstoneproject/Capstonedata/test"

# Görüntü boyutu ve parametreler
image_size = (224, 224)  # Görüntü boyutu
batch_size = 32  # Mini batch boyutu

# 1. Veri yükleme ve artırma
train_datagen = ImageDataGenerator(rescale=1.0/255)  # Normalizasyon
val_datagen = ImageDataGenerator(rescale=1.0/255)    # Normalizasyon
test_datagen = ImageDataGenerator(rescale=1.0/255)   # Normalizasyon

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'  # Çok sınıflı sınıflandırma için
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
    Conv2D(32, (3, 3), activation='relu', input_shape=(512, 512, 3)),
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

# 
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

# 5. Modeli Kaydetme
model.save("cnn_model.keras")

# 6. Eğitim Sonuçlarının Görselleştirilmesi
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')
plt.show()

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')
plt.show()

# 7. Test Sonuçlarının Tahmini
# Test seti tahminleri
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)

# Gerçek sınıfları karşılaştırma
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Karışıklık Matrisi ve Sınıflandırma Raporu
conf_matrix = confusion_matrix(true_classes, predicted_classes)
print("Confusion Matrix:")
print(conf_matrix)

print("Classification Report:")
print(classification_report(true_classes, predicted_classes, target_names=class_labels))
