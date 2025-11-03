import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import json

from net import MnistNN, transform, layers

animation_length = 20


# Load the saved model
model = MnistNN()
model.load_state_dict(torch.load("mnist_nn.pth"))
model.eval()

# Load the MNIST test dataset
mnist_test = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)
test_loader = DataLoader(mnist_test, batch_size=1, shuffle=True)

# Store results for all images
all_results = []

# Run 10 images through the model
print("Processing 10 random images and capturing activations:\n")
with torch.no_grad():
    for i, (image, label) in enumerate(test_loader):
        if i >= animation_length:  # Stop after 10 images
            break

        # Get model prediction with activations
        output, activations, weighted_values = model.forward_with_activations(image)
        _, predicted = torch.max(output, 1)

        # Store results
        result = {
            "image_index": i,
            "actual_label": label.item(),
            "predicted_label": predicted.item(),
            "activations": activations,
            "weighted_values": weighted_values,
            "layers": layers,
        }
        all_results.append(result)

        # Print results
        print(
            f"Image {i+1}: Actual Label = {label.item()}, Model Prediction = {predicted.item()}"
        )

# Save to JSON file
with open("mnist_activations.json", "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\nActivations saved to 'mnist_activations.json'")
print(f"\nStructure of saved data:")
print(f"- activations: Values at each neuron after activation function and bias")
print(f"  - input: 64 values (8x8 flattened input)")
print(f"  - hidden1: 32 values (after ReLU activation)")
print(f"  - hidden2: 16 values (after ReLU activation)")
print(f"  - hidden3: 8 values (after ReLU activation)")
print(f"  - output: 10 values (final logits, no activation)")
print(f"- weighted_values: Values after weights are applied (before activation)")
print(f"  - fc1: 32 values (input -> hidden1)")
print(f"  - fc2: 16 values (hidden1 -> hidden2)")
print(f"  - fc3: 8 values (hidden2 -> hidden3)")
print(f"  - fc4: 10 values (hidden3 -> output)")
