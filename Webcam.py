import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import Execute_ML
import Training_ML  # Import the Training_ML module to get the classes list
import time
import matplotlib.pyplot as plt
import torchvision.models.detection as detection

def update_focus(val):
    global cap
    cap.set(cv2.CAP_PROP_FOCUS, val)

def detect_objects(frame, model, transform):
    # Convert the frame to a PIL image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor_image = transform(image).unsqueeze(0)  # Add batch dimension

    # Perform object detection
    model.eval()
    with torch.no_grad():
        predictions = model(tensor_image)

    return predictions[0]

def draw_bounding_boxes(frame, predictions, threshold=0.5):
    boxes = predictions['boxes']
    scores = predictions['scores']
    for box, score in zip(boxes, scores):
        if score >= threshold:
            x1, y1, x2, y2 = box.int().tolist()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Score: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def inicia_webcam(classes):
    global cap
    # Initialize the webcam
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Erro: Não foi possível abrir a webcam.")
        exit()

    # Activate autofocus
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # 1 to activate, 0 to deactivate

    # Create a window
    cv2.namedWindow('Imagem Capturada')

    # Create a trackbar for focus distance
    cv2.createTrackbar('Focus', 'Imagem Capturada', 0, 100, update_focus)

    # Load a pre-trained object detection model
    model = detection.fasterrcnn_resnet50_fpn(pretrained=True)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    previous_label = None  # Initialize the previous label

    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Erro: Não foi possível ler o frame.")
            break

        # Convert the image to a tensor and resize to 100x100 pixels
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        transform_ml = transforms.Compose([
            transforms.Resize((100, 100)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        tensor_image = transform_ml(pil_image)

        # Pass the complete image to the ML model
        predicted_label = Execute_ML.executa_ml(tensor_image, classes)  # Pass the classes list
        
        # Print the predicted class name only if it changes
        if predicted_label != previous_label:
            print(predicted_label)
            previous_label = predicted_label

        # Perform object detection and draw bounding boxes
        predictions = detect_objects(frame, model, transform)
        draw_bounding_boxes(frame, predictions)

        # Draw bounding boxes and display the results
        cv2.putText(frame, f"Predicted: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Use OpenCV to display the image
        cv2.imshow('Imagem Capturada', frame)

        # Add a delay (e.g., 0.1 seconds)
        time.sleep(0.1)

        # Exit the loop when the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close the windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Get the classes list from the training module
    _, classes, _ = Training_ML.treinamento_ML()
    # Call the function to start the webcam
    inicia_webcam(classes)
