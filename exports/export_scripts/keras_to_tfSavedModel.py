import tensorflow as tf
from tensorflow import keras

# Load the Keras model
model = keras.models.load_model("/path/to/keras_model.keras")

# Export as TensorFlow SavedModel
tf.saved_model.save(model, "/path/to/saved_model_directory")

print("SavedModel export completed.")
