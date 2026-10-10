import os
from torch.utils.tensorboard import SummaryWriter

class ExperimentManager:
    def __init__(self, network_name, lr, batch_size, loss_fn, base_log="experiments/logs",
                 base_checkpoint="experiments/checkpoints", base_result="experiments/results"):
        # log
        self.log_dir = os.path.join(base_log, f"{network_name}_lr{lr}_bs{batch_size}_{type(loss_fn).__name__}")
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)

        # Checkpoint
        self.checkpoint_dir = os.path.join(base_checkpoint, f"{network_name}_lr{lr}_bs{batch_size}_{type(loss_fn).__name__}")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # results
        self.result_dir = os.path.join(base_result, f"{network_name}_lr{lr}_bs{batch_size}_{type(loss_fn).__name__}")
        os.makedirs(self.result_dir, exist_ok=True)

        # metrics
        self.metrics_history = []

    def log_metrics(self, epoch, metrics_dict):
        """
        metrics_dict: {"dice":0.95, "iou":0.9, "assd":1.2, "hd95":1.8, ...}
        """
        self.metrics_history.append({"epoch": epoch, **metrics_dict})

        # TensorBoard
        for k, v in metrics_dict.items():
            self.writer.add_scalar(k, v, epoch)

    def get_latest_metrics(self):
        if self.metrics_history:
            return self.metrics_history[-1]
        return None

    def save_metrics_to_file(self, filename=None):
        if filename is None:
            filename = os.path.join(self.result_dir, "metrics_log.txt")
        with open(filename, "w") as f:
            if self.metrics_history:
                keys = self.metrics_history[0].keys()
                f.write("\t".join(keys) + "\n")
                for m in self.metrics_history:
                    f.write("\t".join([str(m[k]) for k in keys]) + "\n")