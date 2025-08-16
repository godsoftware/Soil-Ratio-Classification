# Soil Ratio Classification

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Results](#results)
- [Data Augmentation](#data-augmentation)
- [Training Configuration](#training-configuration)

## 🌱 Overview

This project focuses on **Soil Ratio Classification** using deep learning techniques to determine the ratio of clay and silt in soil samples. The system utilizes various convolutional neural network (CNN) architectures and transfer learning models to achieve high accuracy in soil composition analysis.

## ✨ Features

- **Multiple CNN Architectures**: Support for various CNN models including custom CNNs, ResNet50, VGG19, and MobileNet
- **Transfer Learning**: Pre-trained models for improved performance
- **Data Augmentation**: Image preprocessing and augmentation techniques
- **Flexible Training**: Configurable epochs, batch sizes, and image dimensions
- **Comprehensive Results**: Detailed training metrics and visualization
- **Cross-Validation**: Train/validation/test split functionality

## 📁 Project Structure

```
Soil-Ratio-Classification/
├── README.md                 # Project documentation
├── scripts/                  # Python scripts for training and testing
│   ├── train_cnn.py         # Main CNN training script
│   ├── cnn100epoch.py       # 100-epoch CNN training
│   ├── Data_augmantet_cnn50-epcıh.py  # Data augmentation CNN
│   ├── mobilnet_epoch50_32.py         # MobileNet training
│   ├── ResNet50_50epoch_32batch_224.py # ResNet50 training
│   ├── vgg19_epoch50.py     # VGG19 training
│   ├── Unet.py              # U-Net architecture
│   ├── test_val_train.py    # Data splitting utility
│   ├── preproces_image_cnn_50epoch.py # Image preprocessing
│   └── rotate.py            # Image rotation utility
└── results/                  # Training results and outputs
    ├── cnn_100epoch_224.224_32batch_88.5/
    ├── cnn_20epoch_297data/
    ├── cnn_50epoch_224.224_32batch/
    ├── mobilnet_50epoch_224.224_32batch/
    ├── ResNET50_50epoch_224.224_32batch/
    ├── vgg19_50epoch_224.224_32batch/
    └── model_epoch50_batch32_cnn_augmanted/
```

## 🚀 Installation

### Prerequisites
- Python 3.7 or higher
- TensorFlow 2.x
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/Soil-Ratio-Classification.git
cd Soil-Ratio-Classification

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install tensorflow opencv-python numpy matplotlib scikit-learn pillow
```

## 💻 Usage

### Basic Training
```bash
# Train a basic CNN model
python scripts/train_cnn.py

# Train with specific parameters
python scripts/cnn100epoch.py
```

### Model-Specific Training
```bash
# Train ResNet50 model
python scripts/ResNet50_50epoch_32batch_224.py

# Train VGG19 model
python scripts/vgg19_epoch50.py

# Train MobileNet model
python scripts/mobilnet_epoch50_32.py
```

### Data Preprocessing
```bash
# Preprocess images
python scripts/preproces_image_cnn_50epoch.py

# Split data into train/validation/test sets
python scripts/test_val_train.py
```

## 🧠 Models

### 1. Custom CNN
- **Architecture**: Convolutional layers with max pooling and dense layers
- **Best Performance**: 88.5% accuracy (100 epochs)
- **Configuration**: 224x224 images, 32 batch size

### 2. ResNet50
- **Architecture**: Pre-trained ResNet50 with transfer learning
- **Configuration**: 50 epochs, 224x224 images, 32 batch size
- **Features**: Skip connections, batch normalization

### 3. VGG19
- **Architecture**: Pre-trained VGG19 with transfer learning
- **Configuration**: 50 epochs, 224x224 images, 32 batch size
- **Features**: Deep architecture, good feature extraction

### 4. MobileNet
- **Architecture**: Lightweight MobileNet for mobile deployment
- **Configuration**: 50 epochs, 224x224 images, 32 batch size
- **Features**: Depth-wise separable convolutions

### 5. U-Net
- **Architecture**: U-Net for semantic segmentation
- **Features**: Encoder-decoder structure, skip connections

## 📊 Results

### Performance Summary
| Model | Epochs | Batch Size | Image Size | Accuracy |
|-------|--------|------------|------------|----------|
| Custom CNN | 100 | 32 | 224x224 | **88.5%** |
| Custom CNN | 50 | 32 | 224x224 | - |
| ResNet50 | 50 | 32 | 224x224 | - |
| VGG19 | 50 | 32 | 224x224 | - |
| MobileNet | 50 | 32 | 224x224 | - |

### Training Results
- **Best Model**: Custom CNN with 100 epochs achieved 88.5% accuracy
- **Data Augmentation**: Improved model generalization
- **Transfer Learning**: Faster convergence with pre-trained models

## 🔧 Data Augmentation

The project includes comprehensive data augmentation techniques:
- **Image Rotation**: Multiple rotation angles
- **Preprocessing**: Normalization, resizing, and color adjustments
- **Batch Processing**: Efficient data loading and augmentation

## 📈 Training Configuration

### Common Parameters
- **Image Dimensions**: 224x224 pixels
- **Batch Size**: 32
- **Epochs**: 20-100 (configurable)
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy, Loss

### Data Split
- **Training**: 70-80%
- **Validation**: 15-20%
- **Testing**: 10-15%
