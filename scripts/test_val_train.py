import os
import shutil
from random import sample

# Giriş ve çıkış klasörlerini tanımlayın
train_folder = "C:/capstoneproject/Capstonedata/train"  # Örneğin: "data/train"
validation_folder = "C:/capstoneproject/Capstonedata/validation"  # Örneğin: "data/val"
test_folder = "C:/capstoneproject/Capstonedata/test"  # Örneğin: "data/test"

# Çıkış klasörlerini oluştur
os.makedirs(validation_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Her sınıf için işlemleri gerçekleştir
for class_name in os.listdir(train_folder):
    class_path = os.path.join(train_folder, class_name)
    if not os.path.isdir(class_path):
        continue  # Eğer bir klasör değilse, atla

    # Sınıfa ait tüm görüntüleri listele
    images = os.listdir(class_path)

    # Eğer görüntü sayısı yeterliyse rastgele 24 fotoğraf seç
    if len(images) >= 50:
        selected_images = sample(images,50)  # Rastgele 24 fotoğraf seç
        val_images = selected_images[:25]  # İlk 12'si validation için
        test_images = selected_images[25:]  # Son 12'si test için
    else:
        print(f"Yetersiz görüntü: {class_name}")
        continue

    # Validation klasörüne kopyala ve ardından train'den sil
    val_class_folder = os.path.join(validation_folder, class_name)
    os.makedirs(val_class_folder, exist_ok=True)
    for img_name in val_images:
        src_path = os.path.join(class_path, img_name)
        dst_path = os.path.join(val_class_folder, img_name)
        shutil.copy(src_path, dst_path)  # Kopyalama
        os.remove(src_path)  # Train'den silme

    # Test klasörüne kopyala ve ardından train'den sil
    test_class_folder = os.path.join(test_folder, class_name)
    os.makedirs(test_class_folder, exist_ok=True)
    for img_name in test_images:
        src_path = os.path.join(class_path, img_name)
        dst_path = os.path.join(test_class_folder, img_name)
        shutil.copy(src_path, dst_path)  # Kopyalama
        os.remove(src_path)  # Train'den silme

    print(f"{class_name} sınıfı için 12 validation ve 12 test fotoğrafı kopyalandı ve train klasöründen silindi.")

print("Tüm sınıflar için işlemler tamamlandı!")
