import cloudinary.uploader
import os

# Cấu hình Cloudinary
cloudinary.config(
    cloud_name=f"{os.getenv('CLOUDINARY_CLOUD_NAME')}",
    api_key=f"{os.getenv('CLOUDINARY_API_KEY')}",
    api_secret=f"{os.getenv('CLOUDINARY_API_SECRET')}",
    secure=True
)

