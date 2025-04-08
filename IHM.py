import sys
from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QPushButton, QWidget
import Training_ML
import Webcam


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sistema de reconhecimento de circunferência (SRC)")
        self.setGeometry(100, 100, 400 , 300)
        self.layout = QVBoxLayout()

        self.label = QLabel("Sistema de reconhecimento de circunferência (SRC)")
        self.label.setStyleSheet("font-size: 20px; font-weight: bold;")
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