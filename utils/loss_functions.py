import torch.nn as nn
import torch
import numpy as np
from torch import einsum
from torch import Tensor
from scipy.ndimage import distance_transform_edt as distance
from typing import Iterable, Set, Tuple
import torch.nn.functional as F
from typing import List
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice

class DiceLoss_BCE(nn.Module):
    def __init__(self, weight=None, alpha = 0.5, size_average=True):
        super(DiceLoss_BCE, self).__init__()
        self.alpha = alpha

    def forward(self, inputs, targets, smooth=1):
        bce_loss = nn.BCEWithLogitsLoss()
        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)
        bce = bce_loss(inputs, targets)
        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return self.alpha*bce + self.alpha*(1-dice)

class JaccardLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(JaccardLoss, self).__init__()


    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()

        jaccard = (intersection + smooth)/(inputs.sum()+targets.sum()-intersection + smooth)
        return  1-jaccard

class JaccardLoss_BCE(nn.Module):
    def __init__(self, weight=None, alpha = 0.5, size_average=True):
        super(JaccardLoss_BCE, self).__init__()
        self.alpha = alpha

    def forward(self, inputs, targets, bce_loss = nn.BCEWithLogitsLoss(), smooth=1):
        
        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)
        bce = bce_loss(inputs, targets)
        intersection = (inputs * targets).sum()

        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        jaccard = (intersection + smooth)/(inputs.sum()+targets.sum()-intersection + smooth)
        return self.alpha*bce + self.alpha*(1-jaccard)

class JBDCLoss(nn.Module):
    def __init__(self, weight = None, alpha = 0.5, size_average = True):
        super(JBDCLoss,self).__init__()
        self.alpha = alpha
    def forward(self, inputs, targets,  bce_loss = nn.BCEWithLogitsLoss(), smooth = 1):
        inputs = torch.sigmoid(inputs)
       
        bce = bce_loss(inputs, targets)
        intersection = (inputs * targets).sum()

        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        jaccard = (intersection + smooth)/(inputs.sum()+targets.sum()-intersection + smooth)
        return self.alpha*bce + (1-self.alpha)*0.5*(1-jaccard) + (1-self.alpha)*0.5*(1-dice)



class BatchSoftDiceLoss(nn.Module):

    def __init__(self,
                 p=1,
                 smooth=1,
                 weight=None,
                 ignore_lb=255):
        super(BatchSoftDiceLoss, self).__init__()
        self.p = p
        self.smooth = smooth
        self.weight = None if weight is None else torch.tensor(weight)
        self.ignore_lb = ignore_lb

    def forward(self, logits, label):
        '''
        args: logits: tensor of shape (N, C, H, W)
        args: label: tensor of shape(N, H, W)
        '''
        # overcome ignored label
        logits = logits.float()
        ignore = label.data.cpu() == self.ignore_lb
        label = label.clone()
        label[ignore] = 0
        lb_one_hot = torch.zeros_like(logits).scatter_(1, label.unsqueeze(1), 1)
        ignore = ignore.nonzero(as_tuple=False)
        _, M = ignore.size()
        a, *b = ignore.chunk(M, dim=1)
        lb_one_hot[[a, torch.arange(lb_one_hot.size(1)).long(), *b]] = 0
        lb_one_hot = lb_one_hot.detach()

        # compute loss
        probs = torch.sigmoid(logits)
        numer = torch.sum((probs * lb_one_hot), dim=(2, 3))
        denom = torch.sum(probs.pow(self.p) + lb_one_hot.pow(self.p), dim=(2, 3))
        if not self.weight is None:
            self.weight = self.weight.to(device)
            numer = numer * self.weight.view(1, -1)
            denom = denom * self.weight.view(1, -1)
        numer = torch.sum(numer)
        denom = torch.sum(denom)
        loss = 1 - (2 * numer + self.smooth) / (denom + self.smooth)
        return loss
    

class BatchSoftDiceLoss_softmax(nn.Module):
    def __init__(self, p=1, smooth=1.0, weight=None, ignore_index=255):
        """
        Dice Loss
        """
        super(BatchSoftDiceLoss_softmax, self).__init__()
        self.p = p
        self.smooth = smooth
        self.ignore_index = ignore_index

        if weight is not None:
            self.register_buffer('weight', torch.tensor(weight, dtype=torch.float32))
        else:
            self.weight = None

    def forward(self, logits, labels):
        """
        Args:
            logits: shape (B, C, H, W)
            labels: shape (B, H, W) with class indices
        """
        B, C, H, W = logits.shape

        mask = (labels != self.ignore_index).unsqueeze(1)  # (B, 1, H, W)
        logits = logits * mask 

        # Softmax 
        probs = F.softmax(logits, dim=1)  # (B, C, H, W)

        # One-hot 
        labels = labels.clone()
        labels[labels == self.ignore_index] = 0
        labels_one_hot = F.one_hot(labels, num_classes=C).permute(0, 3, 1, 2).float()  # (B, C, H, W)
        labels_one_hot = labels_one_hot * mask

        # Dice
        numer = torch.sum((probs * labels_one_hot).pow(self.p), dim=(0, 2, 3))  # per-class
        denom = torch.sum(probs.pow(self.p) + labels_one_hot.pow(self.p), dim=(0, 2, 3))  # per-class

        if self.weight is not None:
            numer = numer * self.weight
            denom = denom * self.weight

        dice = (2 * numer + self.smooth) / (denom + self.smooth)
        loss = 1 - dice.mean()
        return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Focal Loss for multi-class segmentation.

        :param alpha: Tensor of shape [num_classes] to balance different classes, or scalar.
        :param gamma: Focusing parameter (default=2.0).
        :param reduction: 'mean', 'sum', or 'none'.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        :param logits: [B, C, H, W] raw output before softmax
        :param targets: [B, H, W] ground truth with class indices
        """
        num_classes = logits.shape[1]
        device = logits.device
        
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=1)  # [B, C, H, W]
        probs = torch.clamp(probs, min=1e-8, max=1.0)  # Avoid log(0)

        # One-hot encoding of target
        targets_one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2)  # [B, C, H, W]

        # Compute focal loss
        ce_loss = -targets_one_hot * torch.log(probs)  # Cross-entropy loss
        focal_weight = (1 - probs) ** self.gamma  # Focal weight
        focal_loss = focal_weight * ce_loss


        # Apply alpha weighting
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):  # Scalar alpha
                focal_loss *= self.alpha
            elif isinstance(self.alpha, torch.Tensor):  # Per-class alpha
                alpha = self.alpha.view(1, -1, 1, 1)  # Reshape to [1, C, 1, 1]
                focal_loss *= alpha

        # Reduce loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss  # [B, C, H, W]

class Dice_and_FC_loss(nn.Module):
    def __init__(self):
        super(Dice_and_FC_loss, self).__init__()
        alpha = torch.tensor([0.2, 1.0, 1.5])
        self.fc = FocalLoss(alpha=alpha.to(device), gamma=2, reduction='mean')
        # self.dc = BatchSoftDiceLoss(weight=[0,0.5,0.5])
        self.dc =BatchSoftDiceLoss()

    def forward(self, logits, label):
        fc_loss = self.fc(logits, label)
        dc_loss = self.dc(logits, label)

        losses = 0.6  * fc_loss + 0.4 * dc_loss

        return losses

class Dice_and_CE_loss(nn.Module):
    def __init__(self):
        super(Dice_and_CE_loss, self).__init__()

        self.ce = nn.CrossEntropyLoss(ignore_index=0)
        # self.dc = BatchSoftDiceLoss(weight=[0,0.5,0.5])
        self.dc =BatchSoftDiceLoss()

    def forward(self, logits, label):
        ce_loss = self.ce(logits, label)
        dc_loss = self.dc(logits, label)

        losses = 0.3  * ce_loss + 0.7 * dc_loss

        return losses
    
 
# switch between representations
def probs2class(probs: Tensor) -> Tensor:
    b, _, w, h = probs.shape  # type: Tuple[int, int, int, int]
    assert simplex(probs)
 
    res = probs.argmax(dim=1)
    assert res.shape == (b, w, h)
 
    return res
 
 
def probs2one_hot(probs: Tensor) -> Tensor:
    _, C, _, _ = probs.shape
    assert simplex(probs)
 
    res = class2one_hot(probs2class(probs), C)
    assert res.shape == probs.shape
    assert one_hot(res)
 
    return res
 
 
def class2one_hot(seg: Tensor, C: int) -> Tensor:
    if len(seg.shape) == 2:  # Only w, h, used by the dataloader
        seg = seg.unsqueeze(dim=0)
    assert sset(seg, list(range(C)))
 
    b, w, h = seg.shape  # type: Tuple[int, int, int]
 
    res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)
    assert res.shape == (b, C, w, h)
    assert one_hot(res)
 
    return res
 
 
def one_hot2dist(seg: np.ndarray) -> np.ndarray:
    assert one_hot(torch.Tensor(seg), axis=0)
    C: int = len(seg)
 
    res = np.zeros_like(seg)
    # res = res.astype(np.float64)
    for c in range(C):
        posmask = seg[c].astype(np.bool)
 
        if posmask.any():
            negmask = ~posmask
            res[c] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
    return res
 
 
def simplex(t: Tensor, axis=1) -> bool:
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)
 
 
def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])
 
    # Assert utils
 
 
def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())
 
 
def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)
 
 
class SurfaceLoss():
    def __init__(self):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = [1,2]  # https://github.com/LIVIAETS/surface-loss/issues/3
 
    # probs: bcwh, dist_maps: bcwh
    def __call__(self, probs: Tensor, dist_maps: Tensor, _: Tensor) -> Tensor:
        if not simplex(probs):
            print(probs)
        assert simplex(probs)
        assert not one_hot(dist_maps)
 
        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = dist_maps[:, self.idc, ...].type(torch.float32)

        multiplied = einsum("bcwh,bcwh->bcwh", pc, dc)
        # print("Min multiplied value:", multiplied.min().item())
        loss = multiplied.mean()

        # positive_loss = torch.einsum("bcwh,bcwh->bcwh", pc, positive_dc).mean()
        # negative_loss = torch.einsum("bcwh,bcwh->bcwh", pc, negative_dc.abs()).mean()
        # loss = positive_loss + negative_loss 
        return loss
    
        

def soft_dice_score(
    output: torch.Tensor, target: torch.Tensor, smooth: float = 0.0, eps: float = 1e-7, dims=None
) -> torch.Tensor:
    assert output.size() == target.size()
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)
        cardinality = torch.sum(output + target, dim=dims)
    else:
        intersection = torch.sum(output * target)
        cardinality = torch.sum(output + target)
    dice_score = (2.0 * intersection + smooth) / \
        (cardinality + smooth).clamp_min(eps)
    return dice_score

def to_tensor(x, dtype=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, (list, tuple)):
        x = np.ndarray(x)
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x


def build_loss(args):

    if args.loss_fn == "dice":
        return DiceLoss()

    elif args.loss_fn == "jaccard":
        return JaccardLoss()

    elif args.loss_fn == "bce_dice":
        return DiceLoss_BCE(alpha=args.alpha)

    elif args.loss_fn == "bce_jaccard":
        return JaccardLoss_BCE(alpha=args.alpha)

    return BatchSoftDiceLoss()
