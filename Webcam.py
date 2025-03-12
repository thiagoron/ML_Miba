import cv2
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
import Execute_ML  # Importe o seu módulo de ML

def inicia_webcam():
    # Inicialize a webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro: Não foi possível abrir a webcam.")
        exit()

    # Ative o autofoco
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # 1 para ativar, 0 para desativar

    while True:
        ret, frame = cap.read()
        
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
        predicted_label = Execute_ML.executa_ml(tensor_image)  # Chame a função executa_ml

        # Desenhe as caixas delimitadoras e exiba os resultados
        cv2.putText(frame, f"Predicted: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Use matplotlib to display the image
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        plt.title('Imagem Capturada')
        plt.show(block=False)
        plt.pause(0.001)

        # Saia do loop quando a tecla 'q' for pressionada
        if plt.waitforbuttonpress(0.0001) and plt.get_current_fig_manager().canvas.get_tk_widget().focus_get() == 'q':
            break

    # Libere a captura e feche as janelas
    cap.release()
    plt.close()
    cv2.destroyAllWindows()

# Chame a função para iniciar a webcam
inicia_webcam()