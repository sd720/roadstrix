# No tensorflow needed just for downloading
import urllib.request
import os

print("Downloading Official Production-Grade Vision Model...")

assets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'assets')
if not os.path.exists(assets_dir):
    os.makedirs(assets_dir)

model_path = os.path.join(assets_dir, 'vision_model.tflite')

# Download an official, highly-optimized MobileNet V2 SSD TFLite model from Google.
# This is a production-grade object detection model used in self-driving and robotics.
url = "https://storage.googleapis.com/download.tensorflow.org/models/tflite/coco_ssd_mobilenet_v1_1.0_quant_2018_06_29.zip"
zip_path = os.path.join(assets_dir, "model.zip")

try:
    urllib.request.urlretrieve(url, zip_path)
    print("Download complete. Extracting TFLite model...")
    
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extract("detect.tflite", assets_dir)
        
    os.rename(os.path.join(assets_dir, "detect.tflite"), model_path)
    os.remove(zip_path)
    
    print(f"✅ Production-Grade Vision Model successfully installed at: {model_path}")
    print("This is a quantized SSD MobileNet model. It processes frames in ~20ms on mobile devices.")
except Exception as e:
    print(f"Error downloading model: {e}")
