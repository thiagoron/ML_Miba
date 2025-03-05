#Dedicated to run the ML
import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torchvision
import torchvision.transforms as transforms
import Training_ML

# Helper function for inline image display
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        print(classes)
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
    train_loader, classes, transform = Training_ML.treinamento_ML()

    dataiter = iter(train_loader)
    images, labels = next(dataiter)

    # Create a grid from the images and show them
    img_grid = torchvision.utils.make_grid(images)
    matplotlib_imshow(img_grid, one_channel=True)
    print('  '.join(classes[labels[j].item()] for j in range(4)))

    # Load your trained model here
    # Replace YourModelClass with the actual class name of your model
    #model = YourModelClass()
    #model.load_state_dict(torch.load(r'C:\Users\RONZELLADOTH\OneDrive - Miba AG\Área de Trabalho\ML_MIBA\ML_Miba\model.pth'))  # Load the trained model weights
    #model.eval()
    
    # Start webcam inference
    #webcam_inference(model, classes, transform)
    #webcam_inference(model, classes, transform)
