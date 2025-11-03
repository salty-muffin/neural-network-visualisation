import json

# Load the activations data
with open("mnist_activations.json", "r") as f:
    data = json.load(f)

print("Analyzing activation ranges across all images...\n")

# Collect min/max for each layer
layer_stats = {}

for image_data in data:
    activations = image_data["activations"]
    weighted_values = image_data["weighted_values"]

    # Analyze activations
    for layer_name, values in activations.items():
        if layer_name not in layer_stats:
            layer_stats[layer_name] = {
                "activation_min": float("inf"),
                "activation_max": float("-inf"),
            }

        layer_min = min(values)
        layer_max = max(values)
        layer_stats[layer_name]["activation_min"] = min(
            layer_stats[layer_name]["activation_min"], layer_min
        )
        layer_stats[layer_name]["activation_max"] = max(
            layer_stats[layer_name]["activation_max"], layer_max
        )

    # Analyze weighted values
    for weight_key, values in weighted_values.items():
        key_name = f"{weight_key}_weighted"
        if key_name not in layer_stats:
            layer_stats[key_name] = {"min": float("inf"), "max": float("-inf")}

        weight_min = min(values)
        weight_max = max(values)
        layer_stats[key_name]["min"] = min(layer_stats[key_name]["min"], weight_min)
        layer_stats[key_name]["max"] = max(layer_stats[key_name]["max"], weight_max)

# Print results
print("=" * 60)
print("ACTIVATION RANGES (for nodes)")
print("=" * 60)
for layer in ["input", "hidden1", "hidden2", "hidden3", "output"]:
    if layer in layer_stats:
        stats = layer_stats[layer]
        print(
            f"{layer:12s}: [{stats['activation_min']:8.4f}, {stats['activation_max']:8.4f}]"
        )

print("\n" + "=" * 60)
print("WEIGHTED VALUE RANGES (for connections)")
print("=" * 60)
for weight_key in ["fc1_weighted", "fc2_weighted", "fc3_weighted", "fc4_weighted"]:
    if weight_key in layer_stats:
        stats = layer_stats[weight_key]
        print(f"{weight_key:12s}: [{stats['min']:8.4f}, {stats['max']:8.4f}]")

print("\n" + "=" * 60)
print("RECOMMENDED SETTINGS")
print("=" * 60)

# Get overall ranges
all_activation_values = []
all_weighted_values = []

for image_data in data:
    for layer_name, values in image_data["activations"].items():
        all_activation_values.extend(values)
    for weight_key, values in image_data["weighted_values"].items():
        all_weighted_values.extend(values)

overall_act_min = min(all_activation_values)
overall_act_max = max(all_activation_values)
overall_weight_min = min(all_weighted_values)
overall_weight_max = max(all_weighted_values)

print(f"\nFor nodes (activations):")
print(f"  blackpoint = {overall_act_min:.4f}")
print(f"  whitepoint = {overall_act_max:.4f}")

print(f"\nFor connections (weighted values):")
print(f"  blackpoint = {overall_weight_min:.4f}")
print(f"  whitepoint = {overall_weight_max:.4f}")

print(f"\nFor unified visualization (same scale for both):")
print(f"  blackpoint = {min(overall_act_min, overall_weight_min):.4f}")
print(f"  whitepoint = {max(overall_act_max, overall_weight_max):.4f}")

print(f"\nFor input layer visibility (0-1 range for pixels):")
print(f"  blackpoint = 0.0")
print(f"  whitepoint = 1.0")
print(f"  (Good for seeing the image, but other layers may appear very bright)")
