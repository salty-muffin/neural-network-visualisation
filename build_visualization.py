import bpy, json
from mathutils import Vector

with open("/home/bird/git/neural-network-visualisation/trained_activations.json") as f:
    data = json.load(f)

layer_distance = 3
neuron_radius = 0.1
positions = []

# Create neurons
for i, n_neurons in enumerate(data["layers"]):
    for j in range(n_neurons):
        pos = Vector((i * layer_distance, (j - n_neurons / 2) * 0.6, 0))
        bpy.ops.mesh.primitive_uv_sphere_add(radius=neuron_radius, location=pos)
        sphere = bpy.context.object
        sphere.name = f"neuron_{i}_{j}"

# Animate over time
for frame_idx, frame_data in enumerate(data["activations_over_time"]):
    frame = frame_idx * 20  # e.g. 20 frames per input example

    # Input layer
    for j, value in enumerate(frame_data["input"]):
        obj = bpy.data.objects.get(f"neuron_0_{j}")
        if obj:
            obj.scale = (1 + value, 1 + value, 1 + value)
            obj.keyframe_insert(data_path="scale", frame=frame)

    # Hidden layer
    for j, value in enumerate(frame_data["hidden"]):
        obj = bpy.data.objects.get(f"neuron_1_{j}")
        if obj:
            obj.scale = (1 + value, 1 + value, 1 + value)
            obj.keyframe_insert(data_path="scale", frame=frame)

    # Output layer
    for j, value in enumerate(frame_data["output"]):
        obj = bpy.data.objects.get(f"neuron_2_{j}")
        if obj:
            obj.scale = (1 + value, 1 + value, 1 + value)
            obj.keyframe_insert(data_path="scale", frame=frame)
