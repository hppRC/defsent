from torch.optim.lr_scheduler import LambdaLR


class LRPolycy:
    def __init__(self, num_warmup_steps: int) -> None:
        self.num_warmup_steps = num_warmup_steps

    def __call__(self, current_step: int) -> float:
        if current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1.0, self.num_warmup_steps))
        return 1.0


def warmup_scheduler(optimizer, steps_per_epoch: int, epochs: int, warmup_ratio: float):
    num_training_steps = epochs * steps_per_epoch
    num_warmup_steps = num_training_steps * warmup_ratio

    lr_scheduler = LambdaLR(optimizer, lr_lambda=LRPolycy(num_warmup_steps))
    scheduler_config = {
        "scheduler": lr_scheduler,
        "monitor": "val_loss",
        "interval": "step",
        "frequency": 1,
        "strict": True,
    }
    return scheduler_config
