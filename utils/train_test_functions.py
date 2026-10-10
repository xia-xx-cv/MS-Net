import torch
from torchmetrics import MatthewsCorrCoef, JaccardIndex, CohenKappa, Recall, Precision, ROC, Specificity, \
    PrecisionRecallCurve
from collections import defaultdict
import torchvision
from tqdm import tqdm
import os
from utils.loss_functions import *
import cv2
from PIL import Image
from utils.stream_metrics import compute_assd_hd95

loss_b = SurfaceLoss()

metric = defaultdict(list)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
alpha = torch.tensor([0.2, 1.0, 1.5])
focal = FocalLoss(alpha=alpha.to(DEVICE), gamma=2, reduction='mean')


def save_predictions_and_labels(outputs_mask, labels, batch_idx, epoch, save_dir="saved_images5_base"):
    preds = outputs_mask.detach().max(dim=1)[1]
    epoch_dir = os.path.join(save_dir, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)

    preds_mapped = torch.where(preds == 1, 128, torch.where(preds == 2, 0, 255)).cpu().numpy()
    labels_mapped = torch.where(labels == 1, 128, torch.where(labels == 2, 0, 255)).cpu().numpy()

    for i in range(preds_mapped.shape[0]):
        pred_img = Image.fromarray(preds_mapped[i].astype(np.uint8))
        label_img = Image.fromarray(labels_mapped[i].astype(np.uint8))

        pred_img.save(os.path.join(epoch_dir, f"epoch_{epoch}_batch_{batch_idx}_pred_{i}.png"))
        label_img.save(os.path.join(epoch_dir, f"epoch_{epoch}_batch_{batch_idx}_label_{i}.png"))



def save_contours(probs, batch_idx, epoch, y):
    """
    Save contours for the predicted probabilities and overlay the ground truth mask.
    """
    # Convert probabilities to numpy array and get predicted classes
    prob_np = probs.cpu().numpy()  # Move to CPU for numpy conversion
    predicted_classes = np.argmax(prob_np, axis=1)  # Get predicted classes
    # print("Unique predicted classes:", np.unique(predicted_classes))

    # Assuming predicted_classes shape is (batch_size, height, width)
    for i in range(predicted_classes.shape[0]):
        # Get the binary mask for the predicted class (1 for disk, 2 for cup)
        disk_mask = (predicted_classes[i] == 1).astype(np.uint8)  # Assuming class 1 is disk
        cup_mask = (predicted_classes[i] == 2).astype(np.uint8)  # Assuming class 2 is cup

        # Get the ground truth mask for the current sample
        gt_mask = y[i].cpu().numpy()  # Move to CPU and convert to numpy
        gt_disk_mask = (gt_mask == 1).astype(np.uint8)  # Assuming class 1 is disk
        gt_cup_mask = (gt_mask == 2).astype(np.uint8)  # Assuming class 2 is cup

        # Find contours in the masks
        contours_disk, _ = cv2.findContours(disk_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_cup, _ = cv2.findContours(cup_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create an empty image to draw contours
        contour_image = np.zeros((disk_mask.shape[0], disk_mask.shape[1], 3), dtype=np.uint8)

        # Draw contours for the predicted disk and cup
        cv2.drawContours(contour_image, contours_disk, -1, (0, 255, 0), 1)  # Green color for predicted disk
        cv2.drawContours(contour_image, contours_cup, -1, (0, 0, 255), 1)  # Red color for predicted cup

        # Overlay the ground truth contours on the contour image
        gt_contours_disk, _ = cv2.findContours(gt_disk_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        gt_contours_cup, _ = cv2.findContours(gt_cup_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(contour_image, gt_contours_disk, -1, (255, 255, 0), 1)  # Cyan color for ground truth disk
        cv2.drawContours(contour_image, gt_contours_cup, -1, (255, 0, 255), 1)  # Magenta color for ground truth cup

        # Save the contour imag2
        output_dir = f'output5_base/contours/epoch_{epoch}'
        os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist
        output_file_path = os.path.join(output_dir, f'batch_{batch_idx}_sample_{i}.png')
        cv2.imwrite(output_file_path, contour_image)  # Save the contour image directly


def check_gradients_and_weights(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2).item()
            weight_norm = param.data.norm(2).item()
            print(f"Layer: {name}, Gradient Norm: {grad_norm}, Weight Norm: {weight_norm}")


def get_metrics_data():
    return metric

def build_metric_filename(epoch, val_score, extra_metrics=None):
    """
    根据 epoch 和指标生成文件名
    val_score: dict, 包含 "Mean Disc Dice", "Mean Cup Dice", "Disc IoU", "Cup IoU" 等
    extra_metrics: dict, 可选扩展指标
    返回文件名，例如: epoch042_ODDice_0.9700_OCDice_0.9100_ODIoU_0.9500_OCIoU_0.8500.pth
    """
    od_dice = val_score.get("Mean Disc Dice", 0.0)
    oc_dice = val_score.get("Mean Cup Dice", 0.0)
    od_iou = val_score.get("Disc IoU", 0.0)
    oc_iou = val_score.get("Cup IoU", 0.0)

    filename = f"epoch{epoch:03d}_ODDice_{od_dice:.4f}_OCDice_{oc_dice:.4f}_ODIoU_{od_iou:.4f}_OCIoU_{oc_iou:.4f}"

    if extra_metrics:
        for k, v in extra_metrics.items():
            filename += f"_{k}_{v:.4f}"

    filename += ".pth"
    return filename



def save_checkpoint_metric(model, checkpoint_dir, epoch, val_score, extra_metrics=None):
    """
    方式一：按指标命名保存
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filename = build_metric_filename(epoch, val_score, extra_metrics)
    save_path = os.path.join(checkpoint_dir, filename)

    checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'val_score': val_score,
        'extra_metrics': extra_metrics
    }
    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint: {save_path}")
    return save_path

import json

def save_checkpoint_best_last(model, checkpoint_dir, epoch, val_score, current_score, best_score, save_score_file=True):
    """
    save best.pth and last.pth
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # last
    last_path = os.path.join(checkpoint_dir, "last.pth")
    last_checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'val_score': val_score,
        'score': current_score
    }
    torch.save(last_checkpoint, last_path)
    print(f"Saved last model: {last_path}")

    updated_best_score = best_score
    if current_score > best_score:
        # best.pth
        best_path = os.path.join(checkpoint_dir, "best.pth")
        best_checkpoint = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'val_score': val_score,
            'score': current_score
        }
        torch.save(best_checkpoint, best_path)
        print(f"Saved best model: {best_path}")
        updated_best_score = current_score

        # saving scores
        if save_score_file:
            score_info = {
                "epoch": epoch,
                "current_score": current_score,
                "Mean Disc Dice": val_score.get("Mean Disc Dice", 0.0),
                "Mean Cup Dice": val_score.get("Mean Cup Dice", 0.0),
                "Disc IoU": val_score.get("Disc IoU", 0.0),
                "Cup IoU": val_score.get("Cup IoU", 0.0),
                "vCDR MAE": val_score.get("vCDR MAE", 0.0),
                "Score": val_score.get("Score", 0.0),
                "ASSD_Disc": val_score.get("ASSD_Disc", 0.0),
                "ASSD_Cup": val_score.get("ASSD_Cup", 0.0),
                "HD95_Disc": val_score.get("HD95_Disc", 0.0),
                "HD95_Cup": val_score.get("HD95_Cup", 0.0),
            }
            score_file_path = os.path.join(checkpoint_dir, "best_score.json")
            with open(score_file_path, 'w') as f:
                json.dump(score_info, f, indent=4)
            print(f"Saved best score info: {score_file_path}")

    return updated_best_score

def load_checkpoint(checkpoint, model):
    print("=> Loading Checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def save_predictions_as_imgs(
        loader, model, folder, metrics, device="cuda"
):
    metrics.reset()
    model.eval()
    os.makedirs(folder, exist_ok=True)
    for idx, (x, y, c) in enumerate(loader):
        x = x.to(device=device, dtype=torch.float32)
        with torch.no_grad():
            outputs, b = model(x)
            probs = torch.softmax(b, dim=1)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = y.cpu().numpy()
            metrics.update(targets, preds)
            preds = torch.argmax(outputs, dim=1)

            
        for img_idx in range(x.size(0)):
            input_img = x[img_idx].detach().cpu()
            torchvision.utils.save_image(
                input_img, f"{folder}/input_{idx}_{img_idx}.png"
            )

        preds_img = torch.zeros_like(preds, dtype=torch.uint8)
        y_img = torch.zeros_like(y, dtype=torch.uint8)

        # 0: cup -> 0
        preds_img[preds == 2] = 0
        y_img[y == 2] = 0

        # 1: disc -> 128
        preds_img[preds == 1] = 128
        y_img[y == 1] = 128

        # 2: background -> 255
        preds_img[preds == 0] = 255
        y_img[y == 0] = 255
        preds_img = preds_img.to(torch.uint8)
        y_img = y_img.to(torch.uint8)
        torchvision.utils.save_image(
            preds_img.unsqueeze(1).float() / 255.0, f"{folder}/pred_{idx}.png"
        )
        torchvision.utils.save_image(y_img.unsqueeze(1).float() / 255.0, f"{folder}/{idx}.png")

        for class_idx in range(probs.shape[1]):
            prob_map = probs[0, class_idx].detach().cpu().numpy()
            prob_map = (prob_map - prob_map.min()) / (prob_map.max() - prob_map.min()) * 255
            prob_map = prob_map.astype(np.uint8)
            prob_img = Image.fromarray(prob_map)
            prob_img.save(f"{folder}/probs_{idx}_class_{class_idx}.png")

    score = metrics.get_results()
    return score


def compute_metrics(epoch, loader, model, loss_fn, writer, metrics, device="cuda"):
    metrics.reset()

    eps = 1e-8
    total_loss = 0
    total_loss_mask1 = 0
    total_loss_edge = 0

    assd_disc_sum = 0.0
    assd_cup_sum = 0.0
    hd95_disc_sum = 0.0
    hd95_cup_sum = 0.0
    sample_count = 0

    model.eval()
    with torch.no_grad():
        loop = tqdm(loader)
        for batch_idx, (x, y, dist_maps) in enumerate(loop):
            x = x.to(device)
            dist_maps = dist_maps.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.long)
            outputs_mask, outputs_edge = model(x)
            if outputs_edge is not None:
                probs = torch.softmax(outputs_edge, dim=1)
                loss_edge = loss_b(probs, dist_maps, None)
            else:
                loss_edge = torch.tensor(0.0, device=DEVICE, dtype=torch.float32)
            loss_mask1 = loss_fn(outputs_mask, y)
            loss_total = loss_mask1 + 0.5 * loss_edge

            total_loss += loss_total.to(torch.float32)
            total_loss_mask1 += loss_mask1.to(torch.float32)
            total_loss_edge += (0.5 * loss_edge).to(torch.float32)

            # save_contours(probs, batch_idx, epoch, y)
            # save_predictions_and_labels(outputs_mask, y, batch_idx, epoch)

            preds = outputs_mask.detach().max(dim=1)[1].cpu().numpy()
            targets = y.cpu().numpy()
            metrics.update(targets, preds)

            for i in range(targets.shape[0]):
                gt = targets[i]
                pred = preds[i]

                # (class 1)
                gt_disc = (gt == 1)
                pred_disc = (pred == 1)
                assd_d, hd95_d = compute_assd_hd95(gt_disc, pred_disc)
                if not np.isnan(assd_d):
                    assd_disc_sum += assd_d
                    hd95_disc_sum += hd95_d

                # (class 2)
                gt_cup = (gt == 2)
                pred_cup = (pred == 2)
                assd_c, hd95_c = compute_assd_hd95(gt_cup, pred_cup)
                if not np.isnan(assd_c):
                    assd_cup_sum += assd_c
                    hd95_cup_sum += hd95_c

                sample_count += 1

            loop.set_postfix(loss=loss_total.item())

    # avg
    mean_assd_disc = assd_disc_sum / sample_count if sample_count > 0 else 0.0
    mean_hd95_disc = hd95_disc_sum / sample_count if sample_count > 0 else 0.0
    mean_assd_cup = assd_cup_sum / sample_count if sample_count > 0 else 0.0
    mean_hd95_cup = hd95_cup_sum / sample_count if sample_count > 0 else 0.0


    # TensorBoard
    writer.add_scalar('Loss_total/valid', total_loss.item() / len(loader), epoch)
    writer.add_scalar('Loss_mask1/valid', total_loss_mask1.item() / len(loader), epoch)
    writer.add_scalar('Loss_edge/valid', total_loss_edge.item() / len(loader), epoch)
    writer.add_scalar('ASSD/disc', mean_assd_disc, epoch)
    writer.add_scalar('ASSD/cup', mean_assd_cup, epoch)
    writer.add_scalar('HD95/disc', mean_hd95_disc, epoch)
    writer.add_scalar('HD95/cup', mean_hd95_cup, epoch)

    score = metrics.get_results()

    score['ASSD'] = mean_assd
    score['HD95'] = mean_hd95
    score['ASSD_Disc'] = mean_assd_disc
    score['ASSD_Cup'] = mean_assd_cup
    score['HD95_Disc'] = mean_hd95_disc
    score['HD95_Cup'] = mean_hd95_cup

    return score, total_loss_mask1.item() / len(loader)

def train_fn(epoch, loader, model, optimizer, loss_fn, scaler, writer):
    total_loss = 0
    total_loss_mask1 = 0
    total_loss_edge = torch.tensor(0.0, device=DEVICE)
    total_loss_mask2 = 0
    eps = 1e-8
    alpha = 1
    loop = tqdm(loader)
    model.train()
    for batch_idx, (data, targets, dist_maps) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.long().to(device=DEVICE)
        dist_maps = dist_maps.to(device=DEVICE, dtype=torch.float32)
        # forward
        with torch.cuda.amp.autocast():
            # predictions1,predictions2,dist_maps_y = model(data)

            predictions_mask, predictions_edge = model(data)
            if predictions_edge is not None:
                probs = torch.softmax(predictions_edge, dim=1)
                loss_edge = loss_b(probs, dist_maps, None)
            else:
                loss_edge = torch.tensor(0.0, device=DEVICE, dtype=torch.float32)

            loss_mask1 = loss_fn(predictions_mask, targets)
            loss_total = loss_mask1 + 0.5 * loss_edge

            total_loss += (loss_mask1 + 0.5 * loss_edge).to(torch.float32)
            total_loss_mask1 += loss_mask1.to(torch.float32)
            total_loss_edge += (0.5 * loss_edge).to(torch.float32) if predictions_edge is not None else 0.0
            # total_loss_mask2 += loss_mask2.to(torch.float32)
        # backward
        optimizer.zero_grad()
        scaler.scale(loss_total).backward()
        scaler.step(optimizer)
        scaler.update()

        loop.set_postfix(loss=loss_total.item())

    # if epoch<50:
    #   alpha -= 0.01
    print('loss:', total_loss.item() / len(loader))

    writer.add_scalar('Loss_total/train', total_loss.item() / len(loader), epoch)
    writer.add_scalar('Loss_mask1/train', total_loss_mask1.item() / len(loader), epoch)
    writer.add_scalar('Loss_edge/train', total_loss_edge.item() / len(loader), epoch)
    # writer.add_scalar('Loss_mask2/train', total_loss_mask2.item()/len(loader), epoch)


