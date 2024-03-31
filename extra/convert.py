import os
from PIL import Image
import pyheif

def heic_to_jpg(heic_path, jpg_path):
    try:
        # Read HEIC image
        heif_file = pyheif.read(heic_path)
        
        # Convert to RGB format
        image = Image.frombytes(
            heif_file.mode, 
            heif_file.size, 
            heif_file.data,
            "raw",
            heif_file.mode,
            heif_file.stride,
        )
        
        # Save as JPEG
        image.save(jpg_path, "JPEG")
        print(f"Converted {heic_path} to {jpg_path}")
    except Exception as e:
        print(f"Error converting {heic_path}: {str(e)}")

# Directory containing HEIC images
directory = '/home/rohan/Projects/upwork/cat Lora/dataset/march_29/batch 2/'

# Convert each HEIC image to JPEG
for filename in os.listdir(directory):
    if filename.lower().endswith(".heic"):
        heic_path = os.path.join(directory, filename)
        jpg_path = os.path.join(directory, filename.split('.')[0] + '.jpg')
        heic_to_jpg(heic_path, jpg_path)
