import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import Training_ML  # Import the Training_ML module to get the classes list
import matplotlib.pyplot as plt
import torchvision.models.detection as detection
import numpy as np

def update_focus(val):
    global cap
    cap.set(cv2.CAP_PROP_FOCUS, val)

def detect_objects(frame, model, transform):
    # Convert the frame to a PIL image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor_image = transform(image).unsqueeze(0)  # Add batch dimension

    # Perform object detection
    with torch.no_grad():
        predictions = model(tensor_image)

    return predictions[0]

def inicia_webcam(classes):
    global cap
    # Initialize the webcam
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Erro: Não foi possível abrir a webcam.")
        exit()

    # Create a window
    cv2.namedWindow('Imagem Capturada')

    # Create a trackbar for focus distance
    cv2.createTrackbar('Focus', 'Imagem Capturada', 0, 100, update_focus)

    # Load a pre-trained object detection model
    model = detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro: Não foi possível ler o frame.")
            break

        # Detecte o círculo e capture as coordenadas do centro
        center = detect_circle_center(frame)
        if center:
            x, y = center
            cv2.putText(frame, f"Centro: (x: {x}, y: {y})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Exiba a imagem com as coordenadas do centro
        cv2.imshow('Imagem Capturada', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def detect_circle_center(image):
    # Converta a imagem para escala de cinza
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detecte círculos usando a Transformada de Hough
    circles = cv2.HoughCircles(
        gray_image,
        cv2.HOUGH_GRADIENT,
        dp=1.8,
        minDist=30,
        param1=50,
        param2=30,
        minRadius=10,
        maxRadius=30
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            # Desenhe o círculo e o centro na imagem
            cv2.circle(image, (x, y), r, (0, 255, 0), 2)
            cv2.circle(image, (x, y), 2, (0, 0, 255), 3)
            print(f"Círculo detectado no centro: (x: {x:}, y: {y:})")
            return (x, y)
    return None


if __name__ == '__main__':
    # Get the classes list from the training module
    _, classes, _ = Training_ML.treinamento_ML()
    # Call the function to start the webcam
    inicia_webcam(classes)
