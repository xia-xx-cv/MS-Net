# train.py
import argparse
import os
import random
import numpy as np
import torch

from datasets.data_loader import get_loaders
from datasets.transforms import get_train_transform, get_val_transform, get_test_transform
from models.build_model import MODEL_ZOO
from utils.loss_functions import build_loss
from trainer import Trainer
from utils.stream_metrics import StreamSegMetrics
from utils.logger import ExperimentManager

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default=r'/Dataset/refuge')
    parser.add_argument('--epochs', '-e', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', '-B', type=int, default=8)
    parser.add_argument('--loss_fn', type=str, default='jdbc')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw'])
    parser.add_argument('--model', type=str,default='joinedtrans', choices=list(MODEL_ZOO.keys()))
    parser.add_argument("--save_mode", type=str, default="all", choices=["best_last", "metric_name",  "all"])
    args = parser.parse_args()

    seed = 123
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ['PYTHONHASHSEED'] = str(seed)

    train_tf = get_train_transform()
    val_tf = get_val_transform()
    test_tf = get_test_transform()

    train_loader, val_loader, test_loader = get_loaders(
        args,
        os.path.join(args.dir, 'train/crop512/images'),
        os.path.join(args.dir, 'train/crop512/mask'),
        os.path.join(args.dir, 'valid/crop512/images'),
        os.path.join(args.dir, 'valid/crop512/mask'),
        os.path.join(args.dir, 'valid/crop512/images'),
        os.path.join(args.dir, 'valid/crop512/mask'),
        args.batch_size,
        train_tf,
        val_tf,
        test_tf,
        num_workers=0,
        pin_memory=True
    )

    model_class = MODEL_ZOO[args.model]
    model = model_class(n_class=3).to(DEVICE)
    loss_fn = build_loss(args)
    metrics = StreamSegMetrics(3)

    exp_manager = ExperimentManager(args.model, args.lr, args.batch_size, args.loss_fn)

    
    trainer = Trainer(
        args=args,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        metrics=metrics,
        exp_manager=exp_manager,
        device=DEVICE
    )
    trainer.train()

if __name__ == "__main__":
    main()