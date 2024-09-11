import os
import argparse
import pathlib
import json

import torch

from transformers import Trainer, TrainingArguments

from util.train_utils import (
    build_optimizer_scheduler, 
    EarlyStoppingCallbackWithLogging,
    LogCallback
)

import logging


def parse_args() :
    parser = argparse.ArgumentParser()

    # ditributed
    parser.add_argument("--local_rank", type = int)
    parser.add_argument("--fsdp", action = "store_true")
    parser.add_argument("--fsdp_config", type = str, help = "path to fsdp config")
    parser.add_argument("--fsdp_type", type = str, help = "full_shard, auto_wrap, offload settings")
    # To-Do
    # Add DPP arguments.

    # precision
    parser.add_argument("--fp16", action = "store_true")
    parser.add_argument("--bf16", action = "store_true")
    parser.add_argument("--tf32", action = "store_true")

    # paths
    parser.add_argument("--output_dir", type = str, required = True)

    # training args
    parser.add_argument("--lr", type = float, default = 1e-4)
    parser.add_argument("--lr_backbone", type = float, default = 1e-5)
    parser.add_argument("--lr_text_encoder", type = float, default = 5e-5)
    parser.add_argument("--lr_scheduler", type = str, default = "linear")

    parser.add_argument("--train_batch_size", type = int, default = 2, help = "per device")
    parser.add_argument("--eval_batch_size", type = int, default = 2, help = "per device")

    parser.add_argument("--weight_decay", type = float, default = 1e-4)
    parser.add_argument("--warmup_ratio", type = float, default = 0.01)
    parser.add_argument("--clip_max_norm", type = float, default = 0.1)
    parser.add_argument("--gradient_accumulation_steps", type = int, default = 1)

    parser.add_argument("--epochs", type = int, default = 40)
    parser.add_argument("--eval_strategy", type = str, default = "epoch")
    parser.add_argument("--eval_steps", type = int, default = 1)

    # loss co-efficients
    parser.add_argument("--ce_loss_coef", type = float, default = 1.0)
    parser.add_argument("--mask_loss_coef", type = float, default = 1.0)
    parser.add_argument("--dice_loss_coef", type = float, default = 1.0)
    parser.add_argument("--bbox_loss_coef", type = float, default = 5.0)
    parser.add_argument("--giou_loss_coef", type = float, default = 2.0)
    parser.add_argument("--qa_loss_coef", type = float, default = 1.0)
    parser.add_argument("--eos_coef", type = float, default = 0.1, help = "Relative classification weight of the no-object class")
    parser.add_argument("--contrastive_loss_coef", type = float, default = 0.1)
    parser.add_argument("--contrastive_align_loss_coef", type = float, default = 1.0)

    # logging
    parser.add_argument("--logging_strategy", type = str, default = "steps")
    parser.add_argument("--logging_steps", type = int, default = 50)

    args = parser.parse_args()
    return args


if __name__ == "__main__" :
    args = parse_args()

    # set up logging
    os.makedirs(args.output_dir, exist_ok = True)

    logging.basicConfig(
        level = logging.DEBUG,
        handlers = [
            logging.FileHandler(f"{args.output_dir}/{args.experiment_name}.log"),
            logging.StreamHandler()
        ],
        format = "%(asctime)s %(levelname)s: \t%(message)s"
    )

    logging.info("\n" + json.dumps({key : value for key, value in vars(args).items()}, indent = 4) + "\n")

    # not used now
    local_rank = args.local_rank

    # reproducibility
    torch.manual_seed(69)
    torch.use_deterministic_algorithms(True, warn_only = True)

    # build the model
    model, criterion, contrastive_criterion, qa_criterion, weight_dict = build_model(args)

    assert (
        criterion is not None or qa_criterion is not None
    ), "Error: should train either detection or question (or both)."

    # To-Do: Build dataset and collator
    logging.info()
    train_dataset = Dataset()
    val_dataset = Dataset()
    data_collator = Collator()

    # build optimizer and scheduler
    logging.info("")
    args.num_training_steps = (len(train_dataset) + args.train_batch_size - 1) // args.train_batch_size
    optimizer, scheduler = build_optimizer_scheduler(model, args)

    # prepare trainer
    train_args = TrainingArguments()

    # Traaaaaaaaaaain!
    trainer = Trainer(
        model = model,
        train_dataset = train_dataset, 
        eval_dataset = val_dataset, 
        data_collator = data_collator,
        args = train_args,
        callbacks = [
            EarlyStoppingCallbackWithLogging(early_stopping_patience = 3),
            LogCallback()
        ]
    )

    if list(pathlib.Path(args.output_dir).glob("checkpoint-*")) :
        logging.debug(f"Found an existing checkpoint. Resuming training...")
        trainer.train(resume_from_training = True)
    else :
        trainer.train()

    logging.info("Training finished! 🐒")