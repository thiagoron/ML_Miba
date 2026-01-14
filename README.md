# ML_Miba - Circle Detection & Classification System

A comprehensive machine learning project for detecting and classifying circles in real-time using PyTorch and computer vision techniques.

## 📋 Overview

ML_Miba is an intelligent system for real-time circle detection and classification. It combines deep learning with computer vision to identify and classify circular objects through webcam input. The system features:

- **Deep Learning Model**: PyTorch-based convolutional neural network for image classification
- **Real-Time Detection**: Live webcam processing with circle detection using Hough Transform
- **Interactive GUI**: PyQt5 interface for easy training and inference
- **Image Segmentation**: Manual segmentation tools for precise training data preparation
- **Flexible Training**: Train custom models on your own datasets with validation support

## 🎯 Features

- ✅ Real-time circle detection from webcam feed
- ✅ CNN-based image classification (OK/NG categories)
- ✅ Interactive segmentation interface for training data preparation
- ✅ Model training with validation metrics
- ✅ Easy-to-use desktop application interface
- ✅ Focus control and image capture tools
- ✅ CUDA support for GPU acceleration

## 📁 Project Structure

```
ML_Miba/
├── Training_ML.py           # Model training pipeline
├── Execute_ML.py            # Model inference engine
├── Webcam.py                # Webcam capture and circle detection
├── IHM.py                   # GUI interface (PyQt5)
├── Requirements.py          # Package installation utility
├── model.pt                 # Pre-trained model weights
├── Imagens_para_treino/     # Training dataset
│   ├── OK/                  # OK samples
│   └── NG/                  # NG (defective) samples
├── Validacao_treinamento/   # Validation dataset
│   ├── OK/
│   └── NG/
└── README.md
```

## 🏗️ Architecture

### Model Architecture
The system uses a custom CNN with:
- 2 convolutional layers with max pooling
- 3 fully connected layers
- ReLU activation functions
- Cross-entropy loss with Adam optimizer
- Binary classification output (OK/NG)

Input: Grayscale images (100x100 pixels)
Output: Classification probability for each class

### Components
1. **Training Module**: Trains the CNN on labeled datasets with validation
2. **Inference Engine**: Loads trained model and performs predictions
3. **Webcam Module**: Captures live video and detects circles using Hough Transform
4. **GUI Application**: User-friendly interface for all operations

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- Windows/Linux/MacOS
- Webcam device (for live detection)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/thiagoron/ML_Miba.git
cd ML_Miba
```

2. **Install dependencies**
```bash
pip install torch torchvision
pip install PyQt5 opencv-python pillow numpy matplotlib
```

Or install all requirements at once:
```bash
pip install -r requirements.txt
```

### Quick Start

1. **Launch the GUI Application**
```bash
python IHM.py
```

2. **Available Options in GUI**
   - **Training Process**: Train the model on your dataset
   - **Busca Circulo** (Search Circle): Start real-time circle detection from webcam
   - **Close Webcam**: Close any open webcam windows

## 📚 Usage Guide

### Training a New Model

1. Prepare your training data:
   - Place training images in `Imagens_para_treino/OK/` and `Imagens_para_treino/NG/`
   - Place validation images in `Validacao_treinamento/OK/` and `Validacao_treinamento/NG/`

2. Run training via GUI or command line:
```bash
python Training_ML.py
```

3. The script will:
   - Load and preprocess images (grayscale, 100x100 resize)
   - Train the CNN for 20 epochs
   - Display validation accuracy
   - Save the trained model to `model.pt`

### Real-Time Circle Detection

1. Run the application:
```bash
python IHM.py
```

2. Click "Busca Circulo" button to start webcam detection

3. The system will:
   - Capture frames from your webcam
   - Detect circles using Hough Circle Transform
   - Display detected circle centers
   - Press 'Q' to exit

### Model Inference

Use the inference engine directly:
```python
from Execute_ML import executa_ml

# Classify an image
result = executa_ml('path/to/image.png', classes=['OK', 'NG'])
print(f"Classification: {result}")
```

## 🔧 Configuration

### Model Parameters (Training_ML.py)
- **Batch Size**: 4
- **Learning Rate**: 0.001
- **Epochs**: 20
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss
- **Image Size**: 100x100 pixels

### Circle Detection Parameters (Webcam.py)
- **Hough Gradient DP**: 1.8
- **Min Distance**: 30 pixels
- **Param1**: 50
- **Param2**: 30
- **Min Radius**: 10 pixels
- **Max Radius**: 30 pixels

Adjust these values in the source files based on your specific use case.

## 📊 Dataset Format

Training and validation datasets should follow this structure:

```
Imagens_para_treino/
├── OK/
│   ├── image1.png
│   ├── image2.png
│   └── ...
└── NG/
    ├── defect1.png
    ├── defect2.png
    └── ...
```

Each subdirectory represents a class (OK for good/acceptable, NG for not good/defective).

## 🖥️ GPU Support

The system automatically detects and uses CUDA if available:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

If you have an NVIDIA GPU with CUDA support, the training will be significantly faster.

## 📝 Files Description

| File | Purpose |
|------|---------|
| `Training_ML.py` | Model training pipeline with image capture and segmentation |
| `Execute_ML.py` | Inference engine for model predictions |
| `Webcam.py` | Circle detection using OpenCV Hough Transform |
| `IHM.py` | PyQt5 GUI application |
| `Requirements.py` | Offline package installation utility |
| `model.pt` | Pre-trained model weights (PyTorch) |

## 🐛 Troubleshooting

### "Cannot open webcam"
- Ensure your webcam is properly connected
- Check if webcam is not in use by other applications
- Try changing camera index (1 or 0) in `capture_image_from_webcam()` function

### Model accuracy is low
- Add more training samples
- Ensure dataset is balanced between OK and NG classes
- Increase number of epochs
- Adjust learning rate

### CUDA out of memory
- Reduce batch size in `Training_ML.py`
- Use CPU instead by modifying device selection

## 🔄 Workflow

```
1. Prepare Dataset
   ↓
2. Run Training (Training_ML.py)
   ↓
3. Validate Model Performance
   ↓
4. Deploy for Real-Time Detection (Webcam.py)
   ↓
5. Make Predictions (Execute_ML.py)
```

## 📦 Dependencies

- **PyTorch**: Deep learning framework
- **torchvision**: Computer vision utilities
- **OpenCV (cv2)**: Image processing and circle detection
- **Pillow (PIL)**: Image loading and manipulation
- **PyQt5**: GUI framework
- **NumPy**: Numerical computations
- **Matplotlib**: Data visualization

## 💡 Future Enhancements

- [ ] Multi-class classification beyond OK/NG
- [ ] Transfer learning with pre-trained models
- [ ] Real-time confidence display on webcam feed
- [ ] Model quantization for faster inference
- [ ] Batch prediction capability
- [ ] Model versioning and tracking
- [ ] Export to ONNX format

## 📄 License

This project is open source. Feel free to use and modify it for your needs.

## 👥 Author

**Thiago Ronzella**

## 📧 Contact & Support

For issues, questions, or suggestions, please open an issue on the GitHub repository.

## 🙏 Acknowledgments

- PyTorch community for the excellent deep learning framework
- OpenCV for computer vision tools
- PyQt5 for the GUI framework

---

**Last Updated**: January 2026
**Version**: 1.0
