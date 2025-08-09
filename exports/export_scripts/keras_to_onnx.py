import os
import numpy as np
import keras

# Load model
model = keras.models.load_model("/path/to/keras_model.keras")

# Create sample input with correct shape (including batch dimension)
dummy_input = np.random.rand(1, 18, 344, 315, 1).astype(np.float32)

# Run a forward pass with the sample input
model(dummy_input)

# Export model to ONNX format
model.export(
    "/path/to/output_model.onnx",
    format="onnx",
    sample_input=dummy_input
)

print("ONNX export completed.")
