# Dedicated to train the model of ML
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms

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
print(model)

def calculate_output_size(input_size, kernel_size, stride, padding):
    return (input_size - kernel_size + 2 * padding) // stride + 1

def treinamento_ML():
    transform = transforms.Compose(
        [transforms.Grayscale(num_output_channels=1),  # Convert images to grayscale
         transforms.Resize((100, 100)),  # Resize to 100x100 pixels
         transforms.ToTensor(),
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

    # Define the model
    model = YourModelClass().to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    #num_epochs = 10
    #for epoch in range(num_epochs):
    #    model.train()
    #    running_loss = 0.0
    #    for inputs, labels in train_loader:
    #        inputs, labels = inputs.to(device), labels.to(device)
    #        optimizer.zero_grad()
    #        outputs = model(inputs)
    #        loss = criterion(outputs, labels)
    #        loss.backward()
    #        optimizer.step()
    #        running_loss += loss.item()
    #    
    #    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')
    #
    #    # Validation loop
     #   model.eval()
     #   test_loss, correct = 0, 0
     #   with torch.no_grad():
     #       for inputs, labels in validation_loader:
     #           inputs, labels = inputs.to(device), labels.to(device)
     #           outputs = model(inputs)
     #           loss = criterion(outputs, labels)
     #           test_loss += loss.item()
     #           _, predicted = torch.max(outputs, 1)
     #           correct += (predicted == labels).sum().item()
     #   
     #   accuracy = 100 * correct / len(validation_loader.dataset)
     #   print(f'Validation Accuracy: {accuracy:.2f}%, Avg loss: {test_loss/len(validation_loader):.4f}')

    # Save the model
    torch.save(model.state_dict(), r'C:\Users\RONZELLADOTH\OneDrive - Miba AG\Área de Trabalho\ML_MIBA\ML_Miba\model.pt')
    return train_loader, classes, transform