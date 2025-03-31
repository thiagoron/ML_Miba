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
        self.setWindowTitle("ML Front End")
        self.setGeometry(100, 100, 400 , 300)
        self.layout = QVBoxLayout()

        self.label = QLabel("start the process")
        self.layout.addWidget(self.label)

        self.upload_button = QPushButton("Training process")
        self.upload_button.clicked.connect(Training_ML.treinamento_ML)
        self.layout.addWidget(self.upload_button)

        self.upload_button = QPushButton("Close webcam")
        self.upload_button.clicked.connect(Webcam.cv2.destroyAllWindows)
        self.layout.addWidget(self.upload_button)

        self.upload_button = QPushButton("Busca Circulo")
        self.upload_button.clicked.connect(Webcam.inicia_webcam)
        self.layout.addWidget(self.upload_button)

        self.setLayout(self.layout)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())