import os
import shutil
from PIL import Image
import pyexiv2

# Source and destination directories
source_dir = "/home/axelwagner/field_shots_orthomosaic"
destination_dir = "/home/axelwagner/new_images"

# Get a list of image files in the source directory
image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]

# Function to get XMP creation date from metadata (replace with your actual method)
def get_xmp_create_date(image_path):
    # Replace this with your actual code to extract XMP creation date
    metadata = pyexiv2.ImageMetadata(image_path)
    metadata.read()
    val = metadata["Xmp.xmp.CreateDate"].value
    return val

# Create a list of tuples containing (image_path, XMP_create_date)
image_data = [(os.path.join(source_dir, img), get_xmp_create_date(os.path.join(source_dir, img))) for img in image_files]

# Sort images based on XMP creation date
image_data.sort(key=lambda x: x[1])

# Create a new folder if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

# Copy and rename images
for idx, (image_path, _) in enumerate(image_data, start=0):
    _, ext = os.path.splitext(image_path)
    new_filename = f"image_{idx}{ext}"
    new_image_path = os.path.join(destination_dir, new_filename)
    shutil.copy(image_path, new_image_path)
