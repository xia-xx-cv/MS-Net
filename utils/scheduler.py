from torch.optim.lr_scheduler import _LRScheduler, StepLR
# class PolyLR(_LRScheduler):
#     def __init__(self, optimizer, max_iters, power=0.9, last_epoch=-1, min_lr=1e-6):
#         self.power = power
#         self.max_iters = max_iters  # avoid zero lr
#         self.min_lr = min_lr
#         super(PolyLR, self).__init__(optimizer, last_epoch)
    
#     def get_lr(self):
#         return [ max( base_lr * ( 1 - self.last_epoch/self.max_iters )**self.power, self.min_lr)
#                 for base_lr in self.base_lrs]

class PolyLR(_LRScheduler):
    def __init__(self, optimizer, max_epochs, power=0.9, last_epoch=-1, min_lr=1e-8):
        """
        Args:
            optimizer: PyTorch optimizer
            max_epochs: Total number of epochs
            power: The power factor for polynomial decay
            last_epoch: The index of the last epoch
            min_lr: The minimum learning rate
        """
        self.power = power
        self.max_epochs = max_epochs  # Now it's based on epoch
        self.min_lr = min_lr
        super(PolyLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """
        Compute the learning rate for the current epoch using polynomial decay.
        """
        return [
            max(base_lr * (1 - self.last_epoch / self.max_epochs) ** self.power, self.min_lr)
            for base_lr in self.base_lrs
        ]
    

class WarmupPolyLR(_LRScheduler):
    def __init__(self, optimizer, max_epochs, power=0.9, warmup_epochs=5, warmup_start_lr=1e-7, last_epoch=-1):
        self.max_epochs = max_epochs
        self.power = power
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        super(WarmupPolyLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        epoch = self.last_epoch
        if epoch < self.warmup_epochs:
            lr = self.warmup_start_lr + (self.base_lrs[0] - self.warmup_start_lr) * epoch / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            lr = self.base_lrs[0] * (1 - progress) ** self.power
        
        return [lr] * len(self.optimizer.param_groups)