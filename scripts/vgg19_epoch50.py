# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 16:48:49 2024

@author: ozkal
"""

from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np
import os
import cv2
import tensorflow as tf

# GPU Belleği Yönetimi
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Veri seti yolları
train_dir = "C:/capstoneproject/New_Data/data/train"
val_dir = "C:/capstoneproject/New_Data/data/validation"
test_dir = "C:/capstoneproject/New_Data/data/test"

# Görüntü boyutu ve parametreler
image_size = (224, 224)
batch_size = 32

# Veri yükleme
train_datagen = ImageDataGenerator(rescale=1.0 / 255)
val_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical"
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)

# VGG19 Modelini Yükleme
base_model = VGG19(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Base model'in ağırlıklarını dondurma
for layer in base_model.layers:
    layer.trainable = False

# Özelleştirilmiş sınıflandırıcı başlığı ekleme
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)  # Daha büyük bir Dense katman
x = Dropout(0.5)(x)
x = Dense(128, activation="relu")(x)  # İkinci bir Dense katman
x = Dropout(0.3)(x)
output = Dense(len(train_generator.class_indices), activation="softmax")(x)


# Yeni model oluşturma
model = Model(inputs=base_model.input, outputs=output)

# Modelin derlenmesi
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ModelCheckpoint callback'ini tanımlama
checkpoint = ModelCheckpoint(
    filepath="best_model.h5",  # Modeli .h5 formatında kaydediyoruz
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

# Model eğitimi
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=25,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_steps=val_generator.samples // batch_size,
    callbacks=[checkpoint]  # En iyi modeli kaydetmek için callback ekleniyor
)

# Kaydedilen en iyi modeli yükleme
best_model = load_model("best_model.h5")

# En iyi modeli belirli bir klasöre ve özelleştirilmiş bir adla kaydetme
save_dir = "C:/capstoneproject/New_Data/models"  # Modelin kaydedileceği klasör
if not os.path.exists(save_dir):
    os.makedirs(save_dir)  # Klasör yoksa oluştur

model_name = f"best_model_epoch{len(history.history['accuracy'])}_batch{batch_size}_cnn.keras"
save_path = os.path.join(save_dir, model_name)
best_model.save(save_path)
print(f"En iyi model şu dizine kaydedildi: {save_path}")

# Test setinde değerlendirme
test_loss, test_accuracy = best_model.evaluate(test_generator)
print(f"Best Model Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Confusion Matrix
predictions = best_model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

conf_matrix = confusion_matrix(true_classes, predicted_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel("Tahmin Edilen Sınıf")
plt.ylabel("Gerçek Sınıf")
plt.title("Confusion Matrix")
plt.show()

# Classification Report: F1-Score, Precision, Recall
print("\nClassification Report:")
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)

# Eğitim ve Kayıp Sonuçlarının Görselleştirilmesi
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
plt.plot(history.history['loss'], label='Train Loss', color='green')
plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
plt.legend()
plt.title('Loss and Accuracy')
plt.show()

# Tahmin Edilen Görsellerin Gösterimi
def display_all_predicted_images(test_generator, true_classes, predicted_classes, class_labels, limit=50):
    print(f"Toplam {len(test_generator.filenames)} görsel tahmin edildi.")
    plt.figure(figsize=(20, 20))
    for idx in range(min(len(test_generator.filenames), limit)):  # Sınırlı görsel göster
        plt.subplot(5, 10, idx + 1)  # 5x10 grid
        img_path = os.path.join(test_dir, test_generator.filenames[idx])
        img = cv2.imread(img_path)
        if img is None:
            print(f"Görsel yüklenemedi: {img_path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.title(f"Gerçek: {class_labels[true_classes[idx]]}\nTahmin: {class_labels[predicted_classes[idx]]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Tahmin Edilen Görsellerin Gösterimi
display_all_predicted_images(test_generator, true_classes, predicted_classes, class_labels, limit=50)
