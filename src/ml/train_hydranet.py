import tensorflow as tf
import os

print("Building Tesla-Style HydraNet (Multi-Task Vision Model)...")

# 1. Shared Backbone (Feature Extractor)
# In production, this would be MobileNetV2 or ResNet50
input_layer = tf.keras.layers.Input(shape=(224, 224, 3), name='camera_frame')
x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu')(input_layer)
x = tf.keras.layers.MaxPooling2D(2, 2)(x)
x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(x)
x = tf.keras.layers.MaxPooling2D(2, 2)(x)
shared_features = tf.keras.layers.Flatten()(x)

# 2. Multi-Head Architecture (The 3 Hydra Heads)

# Head A: Drivable Space Segmentation (Is it on the road?)
head_a = tf.keras.layers.Dense(32, activation='relu')(shared_features)
output_drivable = tf.keras.layers.Dense(1, activation='sigmoid', name='drivable_space')(head_a)

# Head B: Texture Edge Analysis (Is the asphalt broken?)
head_b = tf.keras.layers.Dense(32, activation='relu')(shared_features)
output_texture = tf.keras.layers.Dense(1, activation='sigmoid', name='broken_texture')(head_b)

# Head C: Depth/Void Estimation (Is it a hole or just a shadow?)
head_c = tf.keras.layers.Dense(32, activation='relu')(shared_features)
output_depth = tf.keras.layers.Dense(1, activation='sigmoid', name='depth_void')(head_c)

# Assemble HydraNet
hydranet = tf.keras.models.Model(
    inputs=input_layer, 
    outputs=[output_drivable, output_texture, output_depth]
)

hydranet.compile(optimizer='adam', loss='binary_crossentropy')

# 3. Export to TFLite
print("Exporting HydraNet to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(hydranet)
tflite_model = converter.convert()

# Ensure the assets directory exists
assets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'assets')
if not os.path.exists(assets_dir):
    os.makedirs(assets_dir)

# Save the model
model_path = os.path.join(assets_dir, 'hydranet.tflite')
with open(model_path, 'wb') as f:
    f.write(tflite_model)

print(f"✅ Tesla-Style HydraNet successfully built and saved to: {model_path}")
print("Outputs: [Drivable Space Mask, Texture Edge Score, Depth Void Score]")
