import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from utils.CLAHE import *
from scipy.ndimage import distance_transform_edt as distance
# from scipy.ndimage import distance_transform_edt as eucl_distance
from torch import Tensor
from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union, cast
from sklearn.model_selection import KFold
from glob import glob
from torch.utils.data import Subset


def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())


def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)


def simplex(t: Tensor, axis=1) -> bool:
    _sum = cast(Tensor, t.sum(axis).type(torch.float32))
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])


def np_class2one_hot(seg: np.ndarray, K: int) -> np.ndarray:
    # print("Np enters")
    """
    Seems to be blocking here when using multi-processing.
    Don't know why, so for now I'll re-implement the same function in numpy
    which should be faster anyhow, but can introduce inconsistencies in the code
    so need to be careful.
    """
    w, h = seg.shape
    seg = seg.astype(np.int64)
    res = np.zeros((K, w, h), dtype=np.int64)
    # np.put_along_axis(res, seg[None, :, :], 1, axis=1)
    for c in range(K):
        res[c, :, :] = (seg == c).astype(np.int64)

    return res


def one_hot2dist(seg: np.ndarray, resolution: Tuple[float, float, float] = None,
                 dtype=None) -> np.ndarray:
    assert one_hot(torch.tensor(seg), axis=0), f"Assertion failed: seg is not one-hot encoded, values: {np.unique(seg)}"
    K: int = len(seg)

    res = np.zeros_like(seg, dtype=dtype)
    for k in range(K):
        posmask = seg[k].astype(bool)

        if posmask.any():
            negmask = ~posmask
            res[k] = distance(negmask, sampling=resolution) * negmask \
                     - (distance(posmask, sampling=resolution) - 1) * posmask
        # The idea is to leave blank the negative classes
        # since this is one-hot encoded, another class will supervise that pixel

    return res


def normalize_dist_maps(dist_maps):
    min_val = dist_maps.min()
    max_val = dist_maps.max()

    if max_val - min_val > 0:
        normalized_dist_maps = (dist_maps - min_val) / (max_val - min_val)
    else:
        normalized_dist_maps = dist_maps

    return normalized_dist_maps


class DatasetLoad(Dataset):
    def __init__(self, args, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        # self.images = os.listdir(image_dir)
        self.images = sorted(os.listdir(image_dir))
        self.args = args

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index])
        mask_path = mask_path.replace('.jpg', '.png')
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 0] = 2
        mask[mask == 128] = 1
        mask[mask == 255] = 0
        # if self.args.train:
        # clahe_transform = CLAHETransform()
        # image = clahe_transform(image)
        one_hot_mask = np_class2one_hot(mask, 3)

        dist_maps = one_hot2dist(one_hot_mask)
        dist_maps = np.transpose(dist_maps, (1, 2, 0))
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask, dist_maps=dist_maps)
            image = augmentations["image"]
            mask = augmentations["mask"]
            dist_maps = augmentations["dist_maps"]

        return image, mask, dist_maps.permute(2, 0, 1)


def gettest_loaders(
        args, test_dir, test_maskdir, batch_size, train_transform, val_transform, test_transform, num_workers=4,
        pin_memory=True):
    test_ds = DatasetLoad(
        args=args,
        image_dir=test_dir,
        mask_dir=test_maskdir,
        transform=test_transform,
    )
   

    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return test_loader


def get_loaders(
        args, train_dir, train_maskdir, val_dir, val_maskdir, test_dir, test_maskdir, batch_size, train_transform,
        val_transform, test_transform, num_workers=4, pin_memory=True):
    train_ds = DatasetLoad(
        args=args,
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = DatasetLoad(
        args=args,
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    test_ds = DatasetLoad(
        args=args,
        image_dir=test_dir,
        mask_dir=test_maskdir,
        transform=test_transform,
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader


from sklearn.model_selection import KFold
from torch.utils.data import Subset


def get_loaders_kfold(args, image_dir, mask_dir, batch_size, train_transform, val_transform, num_workers=4,
                      pin_memory=True, fold=0, n_splits=5):
    image_list = sorted(os.listdir(image_dir))

    # KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = list(kf.split(image_list))
    train_idx, val_idx = splits[fold]

    full_dataset = DatasetLoad(args, image_dir, mask_dir, transform=None)

    train_dataset = Subset(full_dataset, train_idx)
    train_dataset.dataset.transform = train_transform  # 给内部 dataset 设置 transform

    val_dataset = Subset(full_dataset, val_idx)
    val_dataset.dataset.transform = val_transform

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            pin_memory=pin_memory)

    return train_loader, val_loader


from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
import json


def get_loaders_kfold_stratified(args, image_dir, mask_dir, batch_size, train_transform, val_transform,
                                 num_workers=4, pin_memory=True, fold=0, n_splits=5, ratio_json_path=None):
    image_names = sorted(os.listdir(image_dir))
    image_basenames = [os.path.splitext(name)[0] for name in image_names]

    if ratio_json_path is None:
        raise ValueError(" ratio_json_path")
    with open(ratio_json_path, "r") as f:
        ratio_dict = json.load(f)

    for name in image_basenames:
        assert name in ratio_dict, f"{name} not in ratio_dict, pls check JSON files"

    ratios = np.array([ratio_dict[name] for name in image_basenames])
    bin_count = 10
    bins = np.linspace(ratios.min(), ratios.max(), bin_count)
    stratify_labels = np.digitize(ratios, bins)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = list(skf.split(image_names, stratify_labels))
    train_idx, val_idx = splits[fold]

    full_dataset = DatasetLoad(args, image_dir, mask_dir, transform=None)
    train_dataset = Subset(full_dataset, train_idx)
    train_dataset.dataset.transform = train_transform

    val_dataset = Subset(full_dataset, val_idx)
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader



