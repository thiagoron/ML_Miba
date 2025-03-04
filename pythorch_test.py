import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import cv2
import os

# Pytorch tensorboard support
from datetime import datetime
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

def treinamento_ML():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])
    
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

    return train_loader, classes, transform

# Helper function for inline image display
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show() 

def webcam_inference(model, classes, transform):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (32, 32))  # Resize to match the input size of your model
        img = transform(img).unsqueeze(0)  # Apply transformations and add batch dimension

        # Perform inference
        with torch.no_grad():
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)
            label = classes[predicted.item()]

        # Display the result
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Webcam', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    train_loader, classes, transform = treinamento_ML()

    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    # Create a grid from the images and show them
    img_grid = torchvision.utils.make_grid(images)
    matplotlib_imshow(img_grid, one_channel=True)
    print('  '.join(classes[labels[j].item()] for j in range(4)))

    # Load your trained model here
    # Replace YourModelClass with the actual class name of your model
    model = treinamento_ML()
    model.load_state_dict(torch.load('Validacao_treinamento.pth'))
    model.eval()

    # Start webcam inference
    webcam_inference(model, classes, transform)



