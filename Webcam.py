import cv2
import torchvision.transforms as transforms
from PIL import Image
import Execute_ML
import Training_ML  # Import the Training_ML module to get the classes list
import time
import matplotlib.pyplot as plt

def update_focus(val):
    global cap
    cap.set(cv2.CAP_PROP_FOCUS, val)

def inicia_webcam(classes):
    global cap
    # Inicialize a webcam
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Erro: Não foi possível abrir a webcam.")
        exit()

    # Ative o autofoco
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # 1 para ativar, 0 para desativar

    # Create a window
    cv2.namedWindow('Imagem Capturada')

    # Create a trackbar for focus distance
    cv2.createTrackbar('Focus', 'Imagem Capturada', 0, 100, update_focus)

    previous_label = None  # Initialize the previous label

    # Define the region of interest (ROI) coordinates
    roi_x, roi_y, roi_w, roi_h = 200, 200, 300, 300  # Adjust these values as needed

    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Erro: Não foi possível ler o frame.")
            break

        # Draw the ROI rectangle on the frame
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 2)

        # Crop the ROI from the frame
        roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

        # Converta a imagem para um tensor e redimensione para 100x100 pixels
        pil_image = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
        transform = transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        tensor_image = transform(pil_image)

        # Passe a imagem completa para o modelo de ML
        predicted_label = Execute_ML.executa_ml(tensor_image, classes)  # Pass the classes list
        
        # Print the predicted class name only if it changes
        if predicted_label != previous_label:
            print(predicted_label)
            previous_label = predicted_label

        # Desenhe as caixas delimitadoras e exiba os resultados
        cv2.putText(frame, f"Predicted: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Use OpenCV to display the image
        cv2.imshow('Imagem Capturada', frame)

        # Add a delay (e.g., 0.1 seconds)
        time.sleep(0.1)

        # Saia do loop quando a tecla 'q' for pressionada
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libere a captura e feche as janelas
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Get the classes list from the training module
    _, classes, _ = Training_ML.treinamento_ML()
    # Chame a função para iniciar a webcam
    inicia_webcam(classes)