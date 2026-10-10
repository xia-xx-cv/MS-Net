import albumentations as A
from albumentations.pytorch import ToTensorV2

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224

def get_train_transform(dataset='refuge'):
    if dataset == 'refuge':
        mean = [0.6900599, 0.39449272, 0.17383389]
        std = [0.2271874, 0.18058975, 0.14401834]
    elif dataset == 'gamma':
        mean = [0.50791343, 0.29446876, 0.14720915]
        std = [0.21921984, 0.14603596, 0.07648865]
    return A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=45, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25),
        A.Normalize(mean, std),
        ToTensorV2()
    ], additional_targets={"dist_maps":"mask"})

def get_val_transform(dataset='refuge'):
    if dataset == 'refuge':
        mean = [0.6900599, 0.39449272, 0.17383389]
        std = [0.2271874, 0.18058975, 0.14401834]
    elif dataset == 'gamma':
        mean = [0.50791343, 0.29446876, 0.14720915]
        std = [0.21921984, 0.14603596, 0.07648865]
    return A.Compose([
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(mean, std),
        ToTensorV2()
    ], additional_targets={"dist_maps":"mask"})

def get_test_transform():
    return get_val_transform()