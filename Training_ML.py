# Dedicated to train the model of ML
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import cv2
from PIL import Image

# Pytorch tensorboard support
from datetime import datetime
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

class YourModelClass(nn.Module):
    def __init__(self):
        super(YourModelClass, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 22 * 22, 120)  # Adjust input size based on new image dimensions
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 22 * 22)  # Adjust input size based on new image dimensions
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = YourModelClass().to(device)

def calculate_output_size(input_size, kernel_size, stride, padding):
    return (input_size - kernel_size + 2 * padding) // stride + 1

def update_focus(val):
    global cap
    cap.set(cv2.CAP_PROP_FOCUS, val)

def capture_image_from_webcam():
    global cap
    cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print("Erro: Não foi possível abrir a webcam.")
        return None

    # Ative o autofoco
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # 1 para ativar, 0 para desativar

    # Create a window
    cv2.namedWindow('Imagem Capturada')

    # Create a trackbar for focus distance
    cv2.createTrackbar('Focus', 'Imagem Capturada', 0, 100, update_focus)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erro: Não foi possível ler o frame.")
            break

        cv2.imshow('Press "s" to save the image', frame)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            cv2.imwrite('captured_image.png', frame)
            break

    cap.release()
    cv2.destroyAllWindows()
    return 'captured_image.png'

def segment_image(image_path):
    image = cv2.imread(image_path)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    points = []

    def draw_polygon(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(points) > 1:
                cv2.fillPoly(mask, [np.array(points)], 255)
                points.clear()

    cv2.namedWindow('Segment Image')
    cv2.setMouseCallback('Segment Image', draw_polygon)

    while True:
        display_image = cv2.addWeighted(image, 0.7, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), 0.3, 0)
        for point in points:
            cv2.circle(display_image, point, 3, (0, 255, 0), -1)
        if len(points) > 1:
            cv2.polylines(display_image, [np.array(points)], isClosed=False, color=(0, 255, 0), thickness=2)
        cv2.imshow('Segment Image', display_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.imwrite('segmented_mask.png', mask)
    cv2.destroyAllWindows()
    return 'segmented_mask.png'

def treinamento_ML():
    # Capture and segment image
    image_path = capture_image_from_webcam()
    if image_path is None:
        return

    mask_path = segment_image(image_path)

    # Load the captured and segmented images
    captured_image = Image.open(image_path).convert('L')
    segmented_mask = Image.open(mask_path).convert('L')

    transform = transforms.Compose(
        [transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
         transforms.Resize((100, 100)),
         transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])

    tensor_image = transform(captured_image)
    tensor_mask = transform(segmented_mask)

    # Use the tensor_image and tensor_mask for training

    # Define the path to your training and validation data
    train_data_path = r'C:\Users\RONZELLADOTH\OneDrive - Miba AG\Área de Trabalho\ML_MIBA\ML_Miba\Imagens_para_treino'
    validation_data_path = r'C:\Users\RONZELLADOTH\OneDrive - Miba AG\Área de Trabalho\ML_MIBA\ML_Miba\Validacao_treinamento'

    # Create datasets using ImageFolder
    train_set = ImageFolder(root=train_data_path, transform=transform)
    validation_set = ImageFolder(root=validation_data_path, transform=transform)
    
    # Create data loader
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=2)
    validation_loader = DataLoader(validation_set, batch_size=4, shuffle=False, num_workers=2)

    # Class labels
    classes = train_set.classes
    print('Classes:', classes)

    # Report split size
    print('Training set has {} instances'.format(len(train_set)))
    print('Validation set has {} instances'.format(len(validation_set)))

    # Define the model
    model = YourModelClass().to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    if (input("Do you want to train the model? [y/n]: ") == 'y'):
        print("Starting training...")
        num_epochs = 20
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')
        
            # Validation loop
            model.eval()
            test_loss, correct = 0, 0
            with torch.no_grad():
                for inputs, labels in validation_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / len(validation_loader.dataset)
        print(f'Validation Accuracy: {accuracy:.2f}%, Avg loss: {test_loss/len(validation_loader):.4f}')
        print("Training completed.")

        # Save some sample images and their masks
        sample_images, sample_labels = next(iter(validation_loader))
        sample_images, sample_labels = sample_images.to(device), sample_labels.to(device)
        sample_outputs = model(sample_images)
        _, sample_predictions = torch.max(sample_outputs, 1)

        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        for i, ax in enumerate(axes.flatten()):
            if i >= len(sample_images):
                break
            image = sample_images[i].cpu().numpy().transpose((1, 2, 0)).squeeze()
            label = classes[sample_labels[i]]
            prediction = classes[sample_predictions[i]]
            ax.imshow(image, cmap='gray')
            ax.set_title(f"Label: {label}\nPrediction: {prediction}")
            ax.axis('off')
        plt.tight_layout()
        plt.show()

        # Save the model
        torch.save(model.state_dict(), r'model.pt')
    return train_loader, classes, transform
    
if __name__ == '__main__':
    train_loader, classes, transform = treinamento_ML()
