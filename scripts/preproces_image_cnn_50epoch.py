import cv2
import os

def preprocess_and_save_images(input_dir, output_dir):
    """
    Görüntülere kenar tespiti uygular ve işlenmiş görüntüleri belirtilen çıktı dizinine kaydeder.

    Args:
        input_dir (str): Girdi görüntülerinin bulunduğu dizin.
        output_dir (str): İşlenmiş görüntülerin kaydedileceği dizin.
    """
    supported_extensions = ("png", "jpg", "jpeg", "bmp", "tiff")
    skipped_files = []

    for root, dirs, files in os.walk(input_dir):
        for dir_name in dirs:
            input_subdir = os.path.join(root, dir_name)
            output_subdir = os.path.join(output_dir, os.path.relpath(input_subdir, input_dir))
            if not os.path.exists(output_subdir):
                os.makedirs(output_subdir)

        for file in files:
            if file.lower().endswith(supported_extensions):
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_dir, os.path.relpath(root, input_dir), file)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # Görüntüyü yükleme ve kenar tespiti
                img = cv2.imread(input_path)
                if img is not None:
                    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Gri tonlamaya çevirme
                    edges = cv2.Canny(gray_img, 100, 200)  # Kenar tespiti
                    cv2.imwrite(output_path, edges)  # İşlenmiş görüntüyü kaydetme
                else:
                    skipped_files.append(input_path)  # İşlenemeyen dosyaları kaydet

    if skipped_files:
        print("\nİşlenemeyen dosyalar:")
        for skipped in skipped_files:
            print(skipped)

# Girdi ve çıktı dizinleri
train_dir = "C:/capstoneproject/New_Data/data/train"
val_dir = "C:/capstoneproject/New_Data/data/validation"
test_dir = "C:/capstoneproject/New_Data/data/test"

preprocessed_train_dir = "C:/capstoneproject/New_Data/data/preprocessed/train"
preprocessed_val_dir = "C:/capstoneproject/New_Data/data/preprocessed/validation"
preprocessed_test_dir = "C:/capstoneproject/New_Data/data/preprocessed/test"

# Kenar tespiti uygulanmış veri seti oluşturma
preprocess_and_save_images(train_dir, preprocessed_train_dir)
preprocess_and_save_images(val_dir, preprocessed_val_dir)
preprocess_and_save_images(test_dir, preprocessed_test_dir)
