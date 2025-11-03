import torch
import torch.nn as nn


# Define the neural network
class MnistNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 64)  # Input layer to first hidden layer
        self.fc2 = nn.Linear(64, 64)  # First hidden layer to second hidden layer
        self.fc3 = nn.Linear(64, 64)  # Second hidden layer to third hidden layer
        self.fc4 = nn.Linear(64, 10)  # Third hidden layer to output layer

    def forward(self, x):
        x = x.view(-1, 64)  # Flatten the input
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)  # No activation for the output layer (logits)
        return x

    def forward_with_activations(self, x):
        """Forward pass that returns all intermediate values"""
        activations = {}
        weighted_values = {}

        # Input layer
        x = x.view(-1, 64)
        activations["input"] = x.squeeze().tolist()

        # First hidden layer
        z1 = self.fc1(x)  # After weights and bias
        weighted_values["fc1"] = z1.squeeze().tolist()
        a1 = torch.relu(z1)  # After activation
        activations["hidden1"] = a1.squeeze().tolist()

        # Second hidden layer
        z2 = self.fc2(a1)
        weighted_values["fc2"] = z2.squeeze().tolist()
        a2 = torch.relu(z2)
        activations["hidden2"] = a2.squeeze().tolist()

        # Third hidden layer
        z3 = self.fc3(a2)
        weighted_values["fc3"] = z3.squeeze().tolist()
        a3 = torch.relu(z3)
        activations["hidden3"] = a3.squeeze().tolist()

        # Output layer
        z4 = self.fc4(a3)
        weighted_values["fc4"] = z4.squeeze().tolist()
        activations["output"] = z4.squeeze().tolist()

        return z4, activations, weighted_values
