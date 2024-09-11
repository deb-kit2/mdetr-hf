import torch
from dataclasses import dataclass

from transformers import (
    get_linear_schedule_with_warmup,
    EarlyStoppingCallback,
    TrainerCallback
)
from transformers.utils import ModelOutput

from typing import Optional
import logging

logger = logging.getLogger()


@dataclass
class MDETROutput(ModelOutput) :
    """
    Sub-classes the 🤗 Transformers' ModelOutput class to be compatible with MDETR.

    loss : Contains the total weighted loss after one forward pass.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None


class LogCallback(TrainerCallback) :
    """
    Class for logging metrics to the main logger with a file-handler.
    It is recommended not to use this class outside scripts you are not aware of.
    It might look unpleasant to you, but it will get the job done.
    """

    def on_log(self, args, state, control, logs = None, **kwargs) :
        logging.debug(logs)


class EarlyStoppingCallbackWithLogging(EarlyStoppingCallback) :
    """
    Callback for EarlyStoppingWithLogging.
    """

    def __init__(self, early_stopping_patience = 1, early_stopping_threshold = Optional[float] = 0.0) :
        super().__init__(
            early_stopping_patience = early_stopping_patience,
            early_stopping_threshold = early_stopping_threshold
        )

    def on_evaluate(self, args, state, control, metrics, **kwargs) :
        super().on_evaluate(args = args, state = state, control = control, metrics = metrics, **kwargs)

        if control.should_save :
            logging.info(
                f"Epoch {state.epoch}, current patience: {self.early_stopping_patience_counter}/{self.early_stopping_patience}"
            )
    
            if control.should_training_stop and state.epoch < args.num_training_epochs :
                logging.info("EarlyStopping patience reached, stopping early... 🛑")


def build_optimizer_scheduler(model, args) :

    param_dict = [
        {
            "params" : [p for n, p in model.named_parameters() if "backbone" not in n and "text_encoder" not in n and p.requires_grad]
        },
        {
            "params" : [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr" : args.lr_backbone
        },
        {
            "params" : [p for n, p in model.named_parameters() if "text_encoder" in n and p.requires_grad],
            "lr" : args.lr_text_encoder
        }
    ]

    optimizer = torch.optim.AdamW(param_dict, lr = args.lr, weight_decay = args.weight_decay)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps = int(args.num_training_steps * args.warmup_ratio),
        num_training_steps = args.num_training_steps,
        last_epoch = -1
    )

    return optimizer, lr_scheduler
