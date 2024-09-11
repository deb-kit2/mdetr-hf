import os
import argparse
import pathlib
import json

import torch

from models import build_model

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
    parser.add_argument("--exp_name", type = str, default = "training")

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

    # matcher
    parser.add_argument("--set_cost_class", type = float, default = 1, help = "Class coefficient in the matching cost")
    parser.add_argument("--set_cost_bbox", type = float, default = 5, help = "L1 box coefficient in the matching cost")
    parser.add_argument("--set_cost_giou", type = float, default = 2, help = "giou box coefficient in the matching cost")

    # loss co-efficients
    parser.add_argument("--no_aux_loss", dest = "aux_loss", action = "store_false", help = "Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument("--set_loss", type = str, default = "hungarian", choices = ["sequential", "hungarian", "lexicographical"], help = "Type of matching to perform in the loss")
    parser.add_argument("--contrastive_loss", action = "store_true", help = "Whether to add contrastive loss")
    parser.add_argument("--no_contrastive_align_loss", dest = "contrastive_align_loss", action = "store_false", help = "Whether to add contrastive alignment loss")
    parser.add_argument("--contrastive_loss_hdim", type = int, default = 64, help = "Projection head output size before computing normalized temperature-scaled cross entropy loss")

    parser.add_argument("--temperature_NCE", type=float, default=0.07, help="Temperature in the  temperature-scaled cross entropy loss")
    parser.add_argument("--ce_loss_coef", type = float, default = 1.0)
    parser.add_argument("--mask_loss_coef", type = float, default = 1.0)
    parser.add_argument("--dice_loss_coef", type = float, default = 1.0)
    parser.add_argument("--bbox_loss_coef", type = float, default = 5.0)
    parser.add_argument("--giou_loss_coef", type = float, default = 2.0)
    parser.add_argument("--qa_loss_coef", type = float, default = 1.0)
    parser.add_argument("--eos_coef", type = float, default = 0.1, help = "Relative classification weight of the no-object class")
    parser.add_argument("--contrastive_loss_coef", type = float, default = 0.1)
    parser.add_argument("--contrastive_align_loss_coef", type = float, default = 1.0)

    # segmentation
    parser.add_argument("--mask_model", type = str, default = "none", choices = ["none", "smallconv", "v2"], help = "Segmentation head to be used (if None, segmentation will not be trained)")
    parser.add_argument("--remove_difficult", action = "store_true")
    parser.add_argument("--masks", action = "store_true")

    # Model : Text
    parser.add_argument("--freeze_text_encoder", action = "store_true")
    parser.add_argument("--text_encoder_type", type = str, default = "roberta-base", choices = ["roberta-base", "distilroberta-base", "roberta-large"])
    
    # Model : Image
    parser.add_argument("--backbone", type = str, default = "resnet101")
    parser.add_argument("--dilation", action = "store_true", help = "If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument("--position_embedding", type = str, default = "sine", choices = ["sine", "learned"], help = "Type of positional embedding to use on top of the image features")

    # Model : Transformer
    parser.add_argument("--enc_layers", type = int, default = 6, help = "Number of encoding layers in the transformer")
    parser.add_argument("--dec_layers", type = int, default = 6, help = "Number of decoding layers in the transformer")
    parser.add_argument("--dim_feedforward", type = int, default = 2048, help = "Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument("--hidden_dim", type = int, default = 256, help = "Size of the embeddings (dimension of the transformer)")
    parser.add_argument("--dropout", type = float, default = 0.1, help = "Dropout applied in the transformer")
    parser.add_argument("--nheads", type = int, default = 8, help = "Number of attention heads inside the transformer's attentions")
    parser.add_argument("--num_queries", type = int, default = 100, help = "Number of query slots")
    parser.add_argument("--pre_norm", action = "store_true")
    parser.add_argument("--no_pass_pos_and_query", dest = "pass_pos_and_query", action = "store_false", help = "Disables passing the positional encodings to each attention layers")

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

    exit(0)

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