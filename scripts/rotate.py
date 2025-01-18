import cv2
import os

# Giriş ve çıkış klasörleri
input_folder = "C:/capstoneproject/New_Data/data/train/76-100"  # Orijinal fotoğrafların bulunduğu klasör
output_folder = "C:/capstoneproject/New_Data/data/train/76-100"  # İşlenmiş fotoğrafların kaydedileceği klasör
os.makedirs(output_folder, exist_ok=True)  # Çıkış klasörünü oluştur

# İşlem sırası için sayaç
counter = 1

# Giriş klasöründeki tüm dosyaları işle
for img_name in sorted(os.listdir(input_folder)):  # Dosyaları alfabetik sırayla al
    img_path = os.path.join(input_folder, img_name)
    
    # Görüntüyü yükleme
    img = cv2.imread(img_path)
    if img is None:  # Eğer görüntü yüklenemezse atla
        print(f"Görüntü yüklenemedi: {img_name}")
        continue

    # Orijinal görüntüyü sırasıyla kaydetme
    original_name = f"{counter}.jpg"
    cv2.imwrite(os.path.join(output_folder, original_name), img)

    # Yatay çevirme (flipCode = 1)
    flipped_horizontally = cv2.flip(img, 1)
    horizontal_name = f"{counter + 1}.jpg"
    cv2.imwrite(os.path.join(output_folder, horizontal_name), flipped_horizontally)

    # Dikey çevirme (flipCode = 0)
    flipped_vertically = cv2.flip(img, 0)
    vertical_name = f"{counter + 2}.jpg"
    cv2.imwrite(os.path.join(output_folder, vertical_name), flipped_vertically)

    # Sayaç 3 adım ilerler (orijinal + yatay + dikey)
    counter += 3

    print(f"İşlendi: {img_name} -> {original_name}, {horizontal_name}, {vertical_name}")

print("Tüm görüntüler işlendi ve sırayla isimlendirildi!")
