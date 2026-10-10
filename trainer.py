# trainer.py
import os
import torch
import torch.optim as optim
from utils.scheduler import PolyLR, WarmupPolyLR
from torch.utils.tensorboard import SummaryWriter
from utils.train_test_functions import train_fn, compute_metrics, save_checkpoint_metric, save_checkpoint_best_last
from datasets.data_loader import get_loaders_kfold_stratified
from datasets.transforms import get_train_transform, get_val_transform

class Trainer:
    def __init__(self, args, model, train_loader, val_loader, loss_fn, metrics, exp_manager, device):
        self.args = args
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.exp_manager = exp_manager
        self.device = device
        self.best_score = -1

        
        self.optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        if args.optimizer == "adam":
            self.optimizer = optim.Adam(model.parameters(), lr=args.lr)

        self.scheduler = WarmupPolyLR(self.optimizer, max_epochs=args.epochs, power=0.9, warmup_epochs=20)

        self.scaler = torch.cuda.amp.GradScaler()

    def train(self):

        for epoch in range(self.args.epochs):
            print(f"\nEpoch [{epoch + 1}/{self.args.epochs}]")

            # -------------------
            # Train
            # -------------------

            train_fn(
                epoch,
                self.train_loader,
                self.model,
                self.optimizer,
                self.loss_fn,
                self.scaler,
                self.exp_manager.writer,
            )

            # -------------------
            # Validation
            # -------------------

            val_score, vloss = compute_metrics(
                epoch,
                self.val_loader,
                self.model,
                self.loss_fn,
                self.exp_manager.writer,
                self.metrics,
                device=self.device,
            )

            print(self.metrics.to_str(val_score))

            self.scheduler.step()

            # -------------------
            # Record Metrics
            # -------------------

            metrics_dict = {
                "OD_Dice": val_score.get("Mean Disc Dice", 0.0),
                "OC_Dice": val_score.get("Mean Cup Dice", 0.0),
                "OD_IoU": val_score.get("Disc IoU", 0.0),
                "OC_IoU": val_score.get("Cup IoU", 0.0),
                "ASSD": val_score.get("ASSD", 0.0),
                "HD95": val_score.get("HD95", 0.0),
            }

            self.exp_manager.log_metrics(
                epoch,
                metrics_dict
            )

            total_score = val_score["Score"]

            # -------------------
            # Save Model
            # -------------------

            self.save_model(
                epoch,
                val_score,
                total_score
            )

        self.exp_manager.save_metrics_to_file()

    def save_model(self, epoch, val_score, total_score):
        extra_metrics = {
            "AS_OD": val_score.get("ASSD_Disc", 0.0),
            "AS_OC": val_score.get("ASSD_Cup", 0.0),
            "HD_OD": val_score.get("HD95_Disc", 0.0),
            "HD_OC": val_score.get("HD95_Cup", 0.0),
        }

        if self.args.save_mode == "metric_name":
            if total_score > self.best_score:
                self.best_score = total_score
                save_checkpoint_metric(
                    self.model,
                    self.exp_manager.checkpoint_dir,
                    epoch,
                    val_score,
                    extra_metrics
                )
            if epoch == self.args.epochs - 1:
                save_checkpoint_metric(
                    self.model,
                    self.exp_manager.checkpoint_dir,
                    epoch,
                    val_score,
                    extra_metrics
                )

        elif self.args.save_mode == "best_last":
            self.best_score = save_checkpoint_best_last(
                self.model,
                self.exp_manager.checkpoint_dir,
                epoch,
                val_score,
                total_score,
                self.best_score
            )

        elif self.args.save_mode == "all":
            save_checkpoint_metric(
                self.model,
                self.exp_manager.checkpoint_dir,
                epoch,
                val_score,
                extra_metrics
            )


class CrossValTrainer:
    def __init__(self, args, model_class, loss_fn, metrics, device, n_splits=10, ratio_json_path=None):
        self.args = args
        self.model_class = model_class
        self.loss_fn = loss_fn
        self.metrics = metrics
        self.device = device
        self.n_splits = n_splits
        self.ratio_json_path = ratio_json_path

    def train(self):
        for fold in range(self.n_splits):
            print(f"\n--- Training Fold {fold+1}/{self.n_splits} ---")
            model = self.model_class(n_class=3).to(self.device)
            # model = self.model_class().to(self.device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.args.lr)
            if self.args.optimizer == "adam":
                optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)
            scheduler = WarmupPolyLR(optimizer, max_epochs=self.args.epochs, power=0.9, warmup_epochs=20)
            scaler = torch.cuda.amp.GradScaler()

            exp_name = f"{self.args.model}_fold{fold}_lr{self.args.lr}_bs{self.args.batch_size}_{self.args.loss_fn}"
            checkpoint_dir = os.path.join("experiments_gamma", exp_name, "checkpoints")
            log_dir = os.path.join("experiments_gamma", exp_name, "logs")
            os.makedirs(checkpoint_dir, exist_ok=True)
            os.makedirs(log_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=log_dir)

            train_loader, val_loader = get_loaders_kfold_stratified(
                self.args,
                image_dir=os.path.join(self.args.dir, 'crop512/images'),
                mask_dir=os.path.join(self.args.dir, 'crop512/mask'),
                batch_size=self.args.batch_size,
                train_transform=get_train_transform(dataset='gamma'),
                val_transform=get_val_transform(dataset='gamma'),
                num_workers=self.args.num_workers,
                pin_memory=self.args.pin_memory,
                fold=fold,
                n_splits=self.n_splits,
                ratio_json_path=self.ratio_json_path
            )

            best_score = 0.0
            for epoch in range(self.args.epochs):
                print(f"\nEpoch [{epoch+1}/{self.args.epochs}]")
                train_fn(epoch, train_loader, model, optimizer, self.loss_fn, scaler, writer)
                val_score, vloss = compute_metrics(epoch, val_loader, model, self.loss_fn, writer, self.metrics, self.device)
                print(self.metrics.to_str(val_score))
                scheduler.step()

                metrics_dict = {
                    "OD_Dice": val_score.get("Mean Disc Dice", 0.0),
                    "OC_Dice": val_score.get("Mean Cup Dice", 0.0),
                    "OD_IoU": val_score.get("Disc IoU", 0.0),
                    "OC_IoU": val_score.get("Cup IoU", 0.0),
                    "ASSD": val_score.get("ASSD", 0.0),
                    "HD95": val_score.get("HD95", 0.0),
                }
                for name, value in metrics_dict.items():
                    writer.add_scalar(f"valid/{name}", value, epoch)

                total_score = val_score["Score"]

                best_score = self.save_model(
                    model, checkpoint_dir, epoch, val_score, total_score, best_score, fold
                )

            writer.close()
            print(f"Fold {fold+1} completed. Best score: {best_score:.4f}")

    def save_model(self, model, checkpoint_dir, epoch, val_score, total_score, best_score, fold):
        extra_metrics = {
            "ASSD": val_score.get("ASSD", 0.0),
            "HD95": val_score.get("HD95", 0.0),
        }

        if self.args.save_mode == "metric_name":
            if total_score > best_score:
                best_score = total_score
                save_checkpoint_metric(model, checkpoint_dir, epoch, val_score, extra_metrics)
            if epoch == self.args.epochs - 1:
                save_checkpoint_metric(model, checkpoint_dir, epoch, val_score, extra_metrics)

        elif self.args.save_mode == "best_last":
            best_score = save_checkpoint_best_last(
                model, checkpoint_dir, epoch, val_score, total_score, best_score, save_score_file=True
            )

        elif self.args.save_mode == "all":
            save_checkpoint_metric(model, checkpoint_dir, epoch, val_score, extra_metrics)
        return best_score