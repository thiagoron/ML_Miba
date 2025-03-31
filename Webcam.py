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
    with torch.no_grad():
        predictions = model(tensor_image)

    return predictions[0]

#def draw_bounding_boxes(frame, predictions, threshold=0.5):
    boxes = predictions['boxes']
    scores = predictions['scores']
    labels = predictions['labels']  # Class labels

    for box, score, label in zip(boxes, scores, labels):
        if score >= threshold:
            x1, y1, x2, y2 = box.int().tolist()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Score: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def detect_objects_with_model(frame, model, transform):
    predictions = detect_objects(frame, model, transform)
    #draw_bounding_boxes(frame, predictions)
    return frame

def capture_image_from_webcam_with_ml(model, transform):
    global cap
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("Erro: Não foi possível abrir a webcam.")
        return None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro: Não foi possível ler o frame.")
            break

        # Detect objects using the trained model
        frame = detect_objects_with_model(frame, model, transform)

        # Display the frame
        cv2.imshow('Imagem Capturada', frame)

        # Press 's' to save the image
        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite('captured_image.png', frame)
            break

    cap.release()
    cv2.destroyAllWindows()
    return 'captured_image.png'

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

        # Perform object detection
        predictions = detect_objects(frame, model, transform)

        # Draw bounding boxes for detected objects
        #draw_bounding_boxes(frame, predictions)

        # Display the frame
        cv2.imshow('Imagem Capturada', frame)

        # Exit the loop when the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Get the classes list from the training module
    _, classes, _ = Training_ML.treinamento_ML()
    # Call the function to start the webcam
    inicia_webcam(classes)
