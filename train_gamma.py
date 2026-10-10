# train_gamma.py
import argparse
import os
import random
import numpy as np
import torch
from utils.loss_functions import build_loss
from utils.stream_metrics import StreamSegMetrics
from models.build_model import MODEL_ZOO
from trainer import CrossValTrainer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default=r'/Dataset/training_gamma')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--loss_fn', type=str, default='jdbc')
    parser.add_argument('--model', type=str, default='msnet')
    parser.add_argument('--n_splits', type=int, default=10)
    parser.add_argument('--ratio_json', type=str, default=r'RFAUCNxt1/cup_od_ratios.json')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--pin_memory', action='store_true', default=True)
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw'])
    parser.add_argument("--save_mode", type=str, default="best_last", choices=["best_last", "metric_name"])
    args = parser.parse_args()

    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ['PYTHONHASHSEED'] = str(seed)

    loss_fn = build_loss(args)
    metrics = StreamSegMetrics(3)
    model_class = MODEL_ZOO[args.model]

    trainer = CrossValTrainer(
        args=args,
        model_class=model_class,
        loss_fn=loss_fn,
        metrics=metrics,
        device=DEVICE,
        n_splits=args.n_splits,
        ratio_json_path=args.ratio_json
    )
    trainer.train()

if __name__ == "__main__":
    main()