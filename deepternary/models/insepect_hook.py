from typing import Optional, Sequence

from mmengine import print_log
from mmengine.hooks import Hook
from mmengine.model.wrappers import is_model_wrapper
from mmengine.registry import HOOKS
from mmengine.runner import Runner

DATA_BATCH = Optional[Sequence[dict]]


@HOOKS.register_module()
class InsepectHook(Hook):
    """Log model parameter number, trainable parameter number, dataset length, etc."""

    def __init__(self):
        super().__init__()

    def before_run(self, runner: Runner):
        logger = runner.logger
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print_log(f"Total params: {total_params:,}", logger=logger)
        print_log(f"Trainable params: {trainable_params:,}", logger=logger)
        print_log(
            f"Non-trainable params: {total_params - trainable_params:,}", logger=logger
        )
        train_samples = len(runner.train_dataloader.dataset)
        print_log(f"Train samples: {train_samples:,}", logger=logger)
        if hasattr(runner, "val_dataloader"):
            # return self.val_loop.dataloader
            val_samples = len(runner.val_dataloader.dataset)
            print_log(f"Val samples: {val_samples:,}", logger=logger)
        elif hasattr(runner.val_loop, "dataloaders"):  # for multi-val
            val_loaders = runner.val_loop.dataloaders
            val_samples = [len(loader.dataset) for loader in val_loaders]
            print_log(f"There are {len(val_loaders)} val loaders", logger=logger)
            for i, num_samples in enumerate(val_samples):
                print_log(f"Val loader {i}: {num_samples:,}", logger=logger)
