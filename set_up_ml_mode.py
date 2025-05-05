import requests
from tqdm import tqdm

# URL of your shared Dropbox links (make sure they are modified for direct download)
yolov9_for_acne_detection_model = 'https://www.dropbox.com/scl/fi/s4tat0gyr3wiiv9nzmbua/best.pt?rlkey=8lygobguxvubacr0v6520oms4&st=w2mdh0o7&dl=1'
mobileViT_for_acne_severity_model = 'https://www.dropbox.com/scl/fi/vr4vzwjuoom40ooviedsg/best_model.keras?rlkey=sqd5rjtdum31o4bkzhp3du8le&st=9ydkmfwr&dl=1'
convnext_for_skin_classification_model = 'https://www.dropbox.com/scl/fi/9eds3jnzwf68rss2z7l6l/model_convnext_88.pth?rlkey=ksgbscw63yuc8w4fzetzqqahs&e=1&st=mwz3i9qw&dl=1'

# Local paths where you want to save the models
yolov9_for_acne_detection_model_path = './app/ml_models/yolo/weights/best.pt'
mobileViT_for_acne_severity_model_path = './app/ml_models/mobileViT/weights/best_model.keras'
convnext_for_skin_classification_model_path = './app/ml_models/convnext/model_convnext_88.pth'

# Function to download a model from Dropbox with progress bar
def download_model(url, save_path):
    response = requests.get(url, stream=True)

    # Check if the request was successful
    if response.status_code == 200:
        # Get the total file size
        total_size_in_bytes = int(response.headers.get('content-length', 0))

        # Open the file to save the model
        with open(save_path, 'wb') as f:
            # Initialize the progress bar
            with tqdm(total=total_size_in_bytes, unit='B', unit_scale=True, desc=f"Downloading {save_path}") as bar:
                # Write the content to file in chunks
                for data in response.iter_content(chunk_size=1024):
                    f.write(data)
                    bar.update(len(data))  # Update the progress bar
        print(f"Model successfully downloaded and saved as {save_path}")
    else:
        print(f"Failed to download the model. Status code: {response.status_code}")


# Download both models with progress
download_model(yolov9_for_acne_detection_model, yolov9_for_acne_detection_model_path)
download_model(mobileViT_for_acne_severity_model, mobileViT_for_acne_severity_model_path)
download_model(convnext_for_skin_classification_model, convnext_for_skin_classification_model_path)
