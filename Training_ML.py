#Dedicated to train the model of ML
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

    return train_loader, classes, transform, 

