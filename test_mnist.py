import torch
from torchvision import datasets
from torch.utils.data import DataLoader

from net import MnistNN, transform


# Load the saved model
model = MnistNN()
model.load_state_dict(torch.load("mnist_nn.pth"))
model.eval()  # Set the model to evaluation mode

# Load the MNIST test dataset
mnist_test = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
test_loader = DataLoader(mnist_test, batch_size=1, shuffle=True)

# Run 10 images through the model
print("Testing 10 random images:\n")
with torch.no_grad():
    for i, (image, label) in enumerate(test_loader):
        if i >= 10:  # Stop after 10 images
            break

        # Get model prediction
        output = model(image)
        _, predicted = torch.max(output, 1)

        # Print results
        print(
            f"Image {i+1}: Actual Label = {label.item()}, Model Prediction = {predicted.item()}"
        )
