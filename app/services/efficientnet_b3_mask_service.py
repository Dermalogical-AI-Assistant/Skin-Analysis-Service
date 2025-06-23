# import torch
# from torch.optim import AdamW
# from torch.optim.lr_scheduler import CosineAnnealingLR
# from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
# from torch.optim.lr_scheduler import OneCycleLR
#
# OUT_CLASSES = 4
#
# def get_efficientnet_b3_model(out_classes):
#     weights = EfficientNet_B3_Weights.IMAGENET1K_V1
#     model = efficientnet_b3(weights=weights)
#
#     # Sửa đổi conv đầu tiên để nhận 4 kênh (RGB + mask) thay vì 3 kênh
#     original_conv = model.features[0][0]
#     new_conv = nn.Conv2d(
#         in_channels=4,  # 3 RGB + 1 mask
#         out_channels=original_conv.out_channels,
#         kernel_size=original_conv.kernel_size,
#         stride=original_conv.stride,
#         padding=original_conv.padding,
#         bias=original_conv.bias is not None
#     )
#
#     # Copy weights từ pretrained model cho 3 kênh RGB
#     with torch.no_grad():
#         new_conv.weight[:, :3, :, :] = original_conv.weight
#         # Khởi tạo weights cho kênh mask (kênh thứ 4)
#         new_conv.weight[:, 3:4, :, :] = original_conv.weight[:, :1, :, :] * 0.1  # Khởi tạo nhỏ
#         if original_conv.bias is not None:
#             new_conv.bias = original_conv.bias
#
#     # Thay thế conv đầu tiên
#     model.features[0][0] = new_conv
#
#     # Sửa đổi classifier
#     num_ftrs = model.classifier[1].in_features
#     model.classifier = nn.Sequential(
#         nn.Dropout(0.3),
#         nn.Linear(num_ftrs, num_ftrs // 2),
#         nn.SiLU(),
#         nn.Linear(num_ftrs // 2, OUT_CLASSES)
#     )
#     return model
#
#
# model = get_efficientnet_b3_model(OUT_CLASSES).to(device)
#
# def detect_objects_mobileViT(image_bytes: bytes, conf_threshold: float = 0.5):
#     # Đọc ảnh và chuyển sang grayscale
#     image = Image.open(io.BytesIO(image_bytes)).convert("L")  # Grayscale
#     image = image.resize((256, 256))  # Resize về input shape của model
#
#     # Chuyển sang numpy array và normalize
#     image_array = np.array(image).astype("float32") / 255.0
#
#     # Thêm channel axis và batch dimension
#     image_array = np.expand_dims(image_array, axis=-1)  # (256, 256, 1)
#     image_array = np.expand_dims(image_array, axis=0)   # (1, 256, 256, 1)
#
#     # Dự đoán
#     result = model.predict(image_array)
#
#     result_return = {
#         "name": predicted_label,
#         "confidence": float(confidence),
#         "classes": int(predicted_index),
#     }
#
#     return result_return, conf_threshold, classes