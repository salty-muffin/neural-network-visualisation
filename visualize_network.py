import bpy
import json
import math

# Clear existing mesh objects
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)

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


# Calculate grid dimensions for each layer
def get_grid_dimensions(num_neurons):
    """Calculate grid dimensions to arrange neurons in a square"""
    side = math.ceil(math.sqrt(num_neurons))
    return side, side


def create_sphere(location, radius, name):
    """Create a sphere at the given location"""
    bpy.ops.mesh.primitive_ico_sphere_add(
        radius=radius, location=location, subdivisions=2
    )
    sphere = bpy.context.active_object
    sphere.name = name
    return sphere


def create_connection(start_pos, end_pos, radius, name):
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

    return cylinder


# Store node positions
node_positions = {}

# Create nodes for each layer
layers: dict = first_image["layers"]
layer_names = layers.keys()
x_position = 0

for layer_idx, layer_name in enumerate(layer_names):
    num_neurons = layers[layer_name]
    node_positions[layer_name] = []

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
                    create_sphere(pos, node_radius, f"{layer_name}_neuron_{neuron_idx}")

    elif layer_name == "output":
        # Output layer: 1x10 or 2x5 grid
        if num_neurons <= 10:
            # Arrange in a single row
            for i in range(num_neurons):
                y = (i - num_neurons / 2) * output_spacing
                z = 0
                pos = (x_position, y, z)
                node_positions[layer_name].append(pos)
                create_sphere(pos, node_radius, f"{layer_name}_neuron_{i}")
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
                create_sphere(pos, node_radius, f"{layer_name}_neuron_{i}")

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
            create_sphere(pos, node_radius, f"{layer_name}_neuron_{i}")

    x_position += layer_spacing

# Create connections between layers
connection_pairs = [
    ("input", "hidden1"),
    ("hidden1", "hidden2"),
    ("hidden2", "hidden3"),
    ("hidden3", "output"),
]

for start_layer, end_layer in connection_pairs:
    start_positions = node_positions[start_layer]
    end_positions = node_positions[end_layer]

    for i, start_pos in enumerate(start_positions):
        for j, end_pos in enumerate(end_positions):
            print(
                f"{i}/{len(start_positions)}, {j}/{len(end_positions)}, {start_layer} connections"
            )
            create_connection(
                start_pos,
                end_pos,
                connection_radius,
                f"connection_{start_layer}_{i}_to_{end_layer}_{j}",
            )

print(f"Created neural network visualization with {len(layer_names)} layers")
print(f"Total nodes: {sum(len(positions) for positions in node_positions.values())}")
