# Dedicated to run the ML
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms
from PIL import Image  # Import PIL Image
import Training_ML
from Training_ML import YourModelClass

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

# Função para executar o modelo de ML
def executa_ml(image_input, classes):
    # Verifique se a entrada é um caminho de imagem ou um tensor
    if isinstance(image_input, str):
        # Carregar a imagem usando PIL
        image = Image.open(image_input).convert('RGB')
    elif isinstance(image_input, torch.Tensor):
        # Se a entrada já for um tensor, use-a diretamente
        image = image_input
    else:
        raise ValueError("Input should be a file path or a tensor")

    # Redimensione a imagem para 100x100 pixels e converta para escala de cinza
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    if isinstance(image, Image.Image):
        tensor_image = transform(image)
    else:
        tensor_image = image

    # Carregue o modelo treinado
    model = YourModelClass()
    model.load_state_dict(torch.load(r'model.pt'))  # Load the trained model weights
    model.eval()

    # Execute o modelo na imagem tensorizada
    with torch.no_grad():
        outputs = model(tensor_image.unsqueeze(0))  # Adicione uma dimensão para o batch

    # Processar os resultados (ajuste conforme necessário)
    _, predicted = torch.max(outputs, 1)
    
    # Check if the predicted index is within the range of the classes list
    if predicted.item() < len(classes):
        return classes[predicted.item()]
    else:
        return "None"

if __name__ == '__main__':
    train_loader, classes, transform = Training_ML.treinamento_ML()

    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    # Create a grid from the images and show them
    img_grid = torchvision.utils.make_grid(images)
    matplotlib_imshow(img_grid, one_channel=True)
    print('  '.join(classes[labels[j].item()] for j in range(4)))

    # Load your trained model here
    model = YourModelClass()
    model.load_state_dict(torch.load(r'model.pt'))  # Load the trained model weights
    model.eval()
    print(model)
    outputs = model(images)
    print(outputs)


