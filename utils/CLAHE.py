import torch
import numpy as np
import cv2
from torchvision import transforms as T
from PIL import Image

class CLAHETransform:
    def __init__(self, clipLimit=2.0, tileGridSize=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)

    def __call__(self, image):
        image = self.apply_im(image)
        return image


    def apply_im(self, image):
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).numpy().astype(np.uint8)  # 转换为 [H, W, C] 格式
        elif isinstance(image, Image.Image):
            image = np.array(image).astype(np.uint8)

        for i in range(image.shape[2]):
            image[..., i] = self.clahe.apply(image[..., i])

        return image.astype(np.float32)




