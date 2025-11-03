import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# Define the same neural network architecture
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(64, 32)  # Input layer to first hidden layer
        self.fc2 = nn.Linear(32, 16)  # First hidden layer to second hidden layer
        self.fc3 = nn.Linear(16, 8)  # Second hidden layer to third hidden layer
        self.fc4 = nn.Linear(8, 10)  # Third hidden layer to output layer

    def forward(self, x):
        x = x.view(-1, 64)  # Flatten the input
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)  # No activation for the output layer (logits)
        return x


# Load the saved model
model = SimpleNN()
model.load_state_dict(torch.load("mnist_nn.pth"))
model.eval()  # Set the model to evaluation mode

# Define a transform to downscale MNIST to 8x8
transform = transforms.Compose(
    [
        transforms.Resize((8, 8)),  # Resize to 8x8
        transforms.ToTensor(),  # Convert to tensor
    ]
)

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
