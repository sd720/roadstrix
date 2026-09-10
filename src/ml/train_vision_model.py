import tensorflow as tf
import numpy as np
import os

# 1. Define a dummy CNN model architecture (Input: 224x224 RGB Image)
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid') # Output: Pothole Probability (0 to 1)
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 2. Train on dummy synthetic data (just to initialize weights and export architecture)
print("Training Baseline Vision Model (Synthetic Data)...")
dummy_X = np.random.rand(10, 224, 224, 3).astype(np.float32)
dummy_Y = np.random.randint(2, size=(10, 1)).astype(np.float32)

model.fit(dummy_X, dummy_Y, epochs=1, verbose=0)

# 3. Export to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Ensure the assets directory exists
assets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'assets')
if not os.path.exists(assets_dir):
    os.makedirs(assets_dir)

# Save the model
model_path = os.path.join(assets_dir, 'vision_model.tflite')
with open(model_path, 'wb') as f:
    f.write(tflite_model)

print(f"✅ Baseline Vision Model saved to: {model_path}")
print("ℹ️ Note: This is a structural placeholder model. Replace with actual YOLOv8 weights for real accuracy.")
