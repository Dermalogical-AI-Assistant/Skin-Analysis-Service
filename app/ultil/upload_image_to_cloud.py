import numpy as np
from PIL import Image
import os
import uuid
import cloudinary.uploader

def upload_mask_array(mask_array):
    # Đảm bảo mask là uint8 [0, 255]
    mask_array = (mask_array.astype(np.uint8)) * 255

    # Convert về ảnh PIL (grayscale)
    img_pil = Image.fromarray(mask_array)  # mode 'L' - grayscale

    # Tạo thư mục lưu tạm
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(root_dir, "temp_images")
    os.makedirs(save_dir, exist_ok=True)

    # Tạo tên file và lưu
    filename = f"{uuid.uuid4().hex}.png"
    filepath = os.path.join(save_dir, filename)
    img_pil.save(filepath)

    image_url = ''
    try:
        # Upload lên Cloudinary
        upload_result = cloudinary.uploader.upload(filepath, folder="yolov9_detections")
        image_url = upload_result.get("secure_url")
    except Exception as e:
        print(f"Upload failed: {e}")

    # Xoá file tạm
    try:
        os.remove(filepath)
    except Exception as e:
        print(f"Failed to delete file: {e}")

    return image_url
