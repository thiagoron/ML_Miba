import sys
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QPushButton, QFileDialog, QWidget
from PyQt5.QtGui import QPixmap
import torch
import torchvision.transforms as transforms
from PIL import Image
import Training_ML
import Webcam

# Load the trained model
model = Training_ML.YourModelClass()
model.load_state_dict(torch.load('model.pt'))
model.eval()

# Define the transformation
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sistema de reconhecimendo de centro SRC")
        self.setGeometry(100, 100, 400, 300)
        self.layout = QVBoxLayout()

        self.label = QLabel("start the process")
        self.layout.addWidget(self.label)

        self.image_label = QLabel()
        self.layout.addWidget(self.image_label)

        self.upload_button = QPushButton("start webcam")
        self.upload_button.clicked.connect(Training_ML.capture_image_from_webcam)
        self.layout.addWidget(self.upload_button)

        self.upload_button = QPushButton("Close webcam")
        self.upload_button.clicked.connect(Training_ML.cv2.destroyAllWindows)
        self.layout.addWidget(self.upload_button)

        self.setLayout(self.layout)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())