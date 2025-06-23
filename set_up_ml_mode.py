import requests
from tqdm import tqdm
import os 

# URL of your shared Dropbox links
yolov9_for_acne_detection_model = 'https://www.dropbox.com/scl/fi/s4tat0gyr3wiiv9nzmbua/best.pt?rlkey=8lygobguxvubacr0v6520oms4&st=w2mdh0o7&dl=1'
convnext_for_skin_classification_model = 'https://www.dropbox.com/scl/fi/vr2bvxj4g1ej0rup7uplk/model_convnext.pth?rlkey=gkmy0ufqhzhdrfxbuzij295m1&st=rnbobidc&dl=1'
efficientnet_b3_mask = 'https://www.dropbox.com/scl/fi/0iasjhktjs2pq8funn6ob/best.pth?rlkey=lxowps4aihef96y5z25d955fc&st=7o3qfbfw&dl=1'
unetpp = 'https://www.dropbox.com/scl/fi/o5jg8kkh3dcb0zgqy435z/best_model.keras?rlkey=zq49w1t9bir2y3wit4al37egi&st=u8dbxcd8&dl=1'

# Local paths
yolov9_for_acne_detection_model_path = './app/ml_models/yolo/weights/best.pt'
convnext_for_skin_classification_model_path = './app/ml_models/convnext/model_convnext_88.pth'
efficientnet_b3_mask_path = './app/ml_models/efficientnet_b3_mask/weights/best.pth'
unetpp_path = './app/ml_models/unetpp/weights/best_model.keras'

def download_model(url, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Kiểm tra nếu file đã tồn tại thì bỏ qua
    if os.path.exists(save_path):
        print(f"File already exists at {save_path}, skipping download.")
        return

    response = requests.get(url, stream=True)

    if response.status_code == 200:
        total_size_in_bytes = int(response.headers.get('content-length', 0))

        with open(save_path, 'wb') as f:
            with tqdm(total=total_size_in_bytes, unit='B', unit_scale=True, desc=f"Downloading {os.path.basename(save_path)}") as bar:
                for data in response.iter_content(chunk_size=1024):
                    f.write(data)
                    bar.update(len(data))
        print(f"Downloaded and saved: {save_path}")
    else:
        print(f"Failed to download {url}. Status code: {response.status_code}")

def main():
    download_model(yolov9_for_acne_detection_model, yolov9_for_acne_detection_model_path)
    download_model(convnext_for_skin_classification_model, convnext_for_skin_classification_model_path)
    download_model(efficientnet_b3_mask, efficientnet_b3_mask_path)
    download_model(unetpp, unetpp_path)

if __name__ == "__main__":
    main()
