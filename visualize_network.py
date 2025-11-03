import bpy
import json
import math

"""
Neural Network Visualization with Emission Shaders

This script visualizes a neural network in Blender with nodes and connections
that emit light based on their activation values.

Shader Parameters:
- blackpoint: Minimum activation value (maps to 0 emission)
- whitepoint: Maximum activation value (maps to max_emission_strength)
- max_emission_strength: Maximum brightness for nodes/connections at whitepoint

Adjust these values to control the brightness mapping of the visualization.
"""

# Load the activations data
with open(
    "/home/bird/git/neural-network-visualisation/mnist_activations.json", "r"
) as f:
    data = json.load(f)

# Use the first image's data for structure
first_image = data[0]

# Layout parameters
layer_spacing = 5.0  # Distance between layers along X axis
input_spacing = 1
hidden_layer_spacing = 1
output_spacing = 1
node_radius = 0.1  # Radius of sphere nodes
connection_radius = 0.01  # Radius of connection cylinders

# Shader parameters
# Input layer (for image visibility)
input_blackpoint = 0.0  # Minimum activation value for input nodes
input_whitepoint = 1.0  # Maximum activation value for input nodes

# Other layers (hidden + output + connections)
blackpoint = 0  # Minimum activation value mapped to 0 emission
whitepoint = 8.7  # Maximum activation value mapped to max emission

max_emission_strength = 30.0  # Maximum emission strength for white point


# Calculate grid dimensions for each layer
def get_grid_dimensions(num_neurons):
    """Calculate grid dimensions to arrange neurons in a square"""
    side = math.ceil(math.sqrt(num_neurons))
    return side, side


def normalize_activation(value, blackpoint, whitepoint):
    """Normalize activation value between blackpoint and whitepoint"""
    # Clamp value between blackpoint and whitepoint
    clamped = max(blackpoint, min(whitepoint, value))
    # Normalize to 0-1 range
    if whitepoint - blackpoint > 0:
        normalized = (clamped - blackpoint) / (whitepoint - blackpoint)
    else:
        normalized = 0.0
    return normalized


def create_emission_material(name, emission_strength, color=(1, 1, 1)):
    """Create an emission material with given strength"""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear default nodes
    nodes.clear()

    # Create emission shader
    emission = nodes.new(type="ShaderNodeEmission")
    emission.inputs["Color"].default_value = (*color, 1.0)
    emission.inputs["Strength"].default_value = emission_strength

    # Create output node
    output = nodes.new(type="ShaderNodeOutputMaterial")

    # Link emission to output
    links.new(emission.outputs["Emission"], output.inputs["Surface"])

    return mat


def create_sphere(location, radius, name, activation_value=0.0, use_input_range=False):
    """Create a sphere at the given location"""
    bpy.ops.mesh.primitive_ico_sphere_add(
        radius=radius, location=location, subdivisions=2
    )
    sphere = bpy.context.active_object
    sphere.name = name

    # Create and assign emission material
    # Use input-specific range for input layer, standard range for others
    bp = input_blackpoint if use_input_range else blackpoint
    wp = input_whitepoint if use_input_range else whitepoint
    normalized = normalize_activation(activation_value, bp, wp)
    emission_strength = normalized * max_emission_strength
    mat = create_emission_material(f"mat_{name}", emission_strength)
    sphere.data.materials.append(mat)

    return sphere


def create_connection(start_pos, end_pos, radius, name, activation_value=0.0):
    """Create a cylinder connection between two points"""
    # Calculate the vector between points
    dx = end_pos[0] - start_pos[0]
    dy = end_pos[1] - start_pos[1]
    dz = end_pos[2] - start_pos[2]
    dist = math.sqrt(dx**2 + dy**2 + dz**2)

    # Create cylinder
    bpy.ops.mesh.primitive_cylinder_add(
        radius=radius,
        depth=dist,
        vertices=3,
        location=(
            (start_pos[0] + end_pos[0]) / 2,
            (start_pos[1] + end_pos[1]) / 2,
            (start_pos[2] + end_pos[2]) / 2,
        ),
    )
    cylinder = bpy.context.active_object
    cylinder.name = name

    # Rotate cylinder to align with connection
    # Calculate rotation angles
    if dist > 0:
        # Direction vector
        direction = (dx / dist, dy / dist, dz / dist)

        # Calculate rotation using track to constraint approach
        # Align cylinder with the connection vector
        import mathutils

        # Default cylinder points along Z axis
        default_direction = mathutils.Vector((0, 0, 1))
        target_direction = mathutils.Vector(direction)

        rotation = default_direction.rotation_difference(target_direction)
        cylinder.rotation_euler = rotation.to_euler()

    # Create and assign emission material
    normalized = normalize_activation(activation_value, blackpoint, whitepoint)
    emission_strength = normalized * max_emission_strength
    mat = create_emission_material(f"mat_{name}", emission_strength)
    cylinder.data.materials.append(mat)

    return cylinder


# Store node positions
node_positions = {}

# Get activations data
activations = first_image["activations"]
weighted_values = first_image["weighted_values"]

# Create nodes for each layer
layers: dict = first_image["layers"]
layer_names = layers.keys()
x_position = 0

for layer_idx, layer_name in enumerate(layer_names):
    num_neurons = layers[layer_name]
    node_positions[layer_name] = []

    # Get activation values for this layer
    layer_activations = activations.get(layer_name, [0] * num_neurons)

    if layer_name == "input":
        # Input layer: 8x8 grid
        grid_size = get_grid_dimensions(layers["input"])[0]
        for i in range(grid_size):
            for j in range(grid_size):
                neuron_idx = i * grid_size + j
                if neuron_idx < num_neurons:
                    y = (j - grid_size / 2) * input_spacing
                    z = (i - grid_size / 2) * input_spacing
                    pos = (x_position, y, z)
                    node_positions[layer_name].append(pos)
                    activation_val = (
                        layer_activations[neuron_idx]
                        if neuron_idx < len(layer_activations)
                        else 0.0
                    )
                    create_sphere(
                        pos,
                        node_radius,
                        f"{layer_name}_neuron_{neuron_idx}",
                        activation_val,
                        use_input_range=True,  # Use input-specific range
                    )

    elif layer_name == "output":
        # Output layer: 1x10 or 2x5 grid
        if num_neurons <= 10:
            # Arrange in a single row
            for i in range(num_neurons):
                y = (i - num_neurons / 2) * output_spacing
                z = 0
                pos = (x_position, y, z)
                node_positions[layer_name].append(pos)
                activation_val = (
                    layer_activations[i] if i < len(layer_activations) else 0.0
                )
                create_sphere(
                    pos, node_radius, f"{layer_name}_neuron_{i}", activation_val
                )
        else:
            # Arrange in a grid
            side_y, side_z = get_grid_dimensions(num_neurons)
            for i in range(num_neurons):
                row = i // side_y
                col = i % side_y
                y = (col - side_y / 2) * output_spacing
                z = (row - side_z / 2) * output_spacing
                pos = (x_position, y, z)
                node_positions[layer_name].append(pos)
                activation_val = (
                    layer_activations[i] if i < len(layer_activations) else 0.0
                )
                create_sphere(
                    pos, node_radius, f"{layer_name}_neuron_{i}", activation_val
                )

    else:
        # Hidden layers: square grid
        side = math.ceil(math.sqrt(num_neurons))
        spacing = 0.2
        for i in range(num_neurons):
            row = i // side
            col = i % side
            y = (col - side / 2) * hidden_layer_spacing
            z = (row - side / 2) * hidden_layer_spacing
            pos = (x_position, y, z)
            node_positions[layer_name].append(pos)
            activation_val = layer_activations[i] if i < len(layer_activations) else 0.0
            create_sphere(pos, node_radius, f"{layer_name}_neuron_{i}", activation_val)

    x_position += layer_spacing

# Create connections between layers
connection_pairs = [
    ("input", "hidden1", "fc1"),
    ("hidden1", "hidden2", "fc2"),
    ("hidden2", "hidden3", "fc3"),
    ("hidden3", "output", "fc4"),
]

for start_layer, end_layer, weight_key in connection_pairs:
    start_positions = node_positions[start_layer]
    end_positions = node_positions[end_layer]

    # Get weighted values for this layer connection
    layer_weighted_values = weighted_values.get(weight_key, [0] * len(end_positions))

    for i, start_pos in enumerate(start_positions):
        for j, end_pos in enumerate(end_positions):
            print(
                f"{i}/{len(start_positions)}, {j}/{len(end_positions)}, {start_layer} connections"
            )
            # Use the weighted value of the target neuron for this connection
            # This represents the contribution to that neuron
            weight_val = (
                layer_weighted_values[j] if j < len(layer_weighted_values) else 0.0
            )
            create_connection(
                start_pos,
                end_pos,
                connection_radius,
                f"connection_{start_layer}_{i}_to_{end_layer}_{j}",
                weight_val,
            )

print(f"Created neural network visualization with {len(layer_names)} layers")
print(f"Total nodes: {sum(len(positions) for positions in node_positions.values())}")
