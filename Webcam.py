import cv2
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import Execute_ML
import Training_ML  # Import the Training_ML module to get the classes list
import time

def inicia_webcam():
    # Inicialize a webcam
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Erro: Não foi possível abrir a webcam.")
        exit()

    # Ative o autofoco
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # 1 para ativar, 0 para desativar

    # Get the classes list from the training module
    _, classes, _ = Training_ML.treinamento_ML()

    while True:
        ret, frame = cap.read()
        predicted_label = None
        
        if not ret:
            print("Erro: Não foi possível ler o frame.")
            break

        # Converta a imagem para um tensor e redimensione para 100x100 pixels
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        transform = transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        tensor_image = transform(pil_image)

        # Passe a imagem completa para o modelo de ML
        predicted_label = Execute_ML.executa_ml(tensor_image,classes)  # Pass the classes list
        print(predicted_label)  # Print the predicted class name
        

        # Desenhe as caixas delimitadoras e exiba os resultados
        #cv2.putText(frame, f"Predicted: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Use OpenCV to display the image
        cv2.imshow('Imagem Capturada', frame)
        time.sleep(0.1)

        # Saia do loop quando a tecla 'q' for pressionada
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    # Libere a captura e feche as janelas
    cap.release()
    cv2.destroyAllWindows()

# Chame a função para iniciar a webcam
inicia_webcam()