# FastAPI Project with YOLOv9 and Faster R-CNN

This is a Skin Analysis Service for image detection using two popular deep learning models: **YOLOv9** and **Faster R-CNN**. The application provides APIs to perform object detection on images, leveraging these models to identify objects with high accuracy.

## Features

- **YOLOv9**: Real-time object detection with YOLOv9.
- **Faster R-CNN**: Object detection using the Faster R-CNN model.
- **MobileViT**: Classification model for image classification tasks.
- **RESTful API**: Built with FastAPI, providing endpoints for object detection.
## Requirements

To run this project, make sure you have the following installed:

- Python 3.11
- Pip (for installing dependencies)
- A virtual environment (recommended)

## Setup

Follow these steps to get your development environment up and running:

### 1. Clone the repository

```bash
git@github.com:Dermalogical-AI-Assistant/Skin-Analysis-Service.git
cd Skin-Analysis-Service
```

### 2. Create a virtual environment

```bash
python3.11 -m venv venv
```
- if you don't have python3.11 you can run the following command to install it
```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11 python3.11-ven
```

#### 3. Activate the virtual environment
- On Windows:
```bash
venv\Scripts\activate
```
- On macOS/Linux:
```bash
source venv/bin/activate
```
### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Clone Models
```bash
python3 ./setup_ml_models.py
```

### 6. Configure Environment Variables
Create a `.env` file in the root directory of the project and add the following variables:

```bash
touch ..env
```
### 7. Install dependence
```bash
    pip install -r requirements.txt
```

### 8. Run the application

```bash
uvicorn main:app --host
```

