import numpy as np
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from tqdm import tqdm
import transformers

from transformers import Adafactor, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from transformers import RoFormerConfig, BatchEncoding
import os
import json
import logging
from datetime import datetime

import datasets

from accelerate import Accelerator
from accelerate.utils import LoggerType
from accelerate.logging import get_logger

from dataset import load_dataset
from tokenizer import CebinTokenizer
from models import CEBinPairEncoder

from helper import (
    get_tflops,
    model_size_formatter,
    set_cpu_maximum_parallelism,
    get_model_size,
)

from functools import partial
from time import time
import argparse
import random
import re
from config import FEATURE_PATTERN, DATASET_FEATURES
# from azureml.core.run import Run
# az_logger = Run.get_context()
logger = get_logger(__name__)

class TripletLoss(nn.Module):
    def __init__(self,margin):
        super().__init__()
        self.margin=margin

    def forward(self, good_logits, bad_logits):        
        loss=(self.margin-(good_logits-bad_logits)).clamp(min=1e-6).mean()
        return loss

def pick_features(keywords, features):
    if keywords == "all":
        return None
    return set(
        [feature for feature in features if any(key in feature for key in keywords)]
    )


def collate_fn(batch):
    positive_features = {f: [] for f in ["input_ids", "attention_mask", "token_type_ids"]}
    negative_features = {f: [] for f in ["input_ids", "attention_mask", "token_type_ids"]}
    max_length = 1024

    def concate_pair(func1, func2):
        concate_features = {f: [] for f in ["input_ids", "attention_mask", "token_type_ids"]}
        truncate_len_1 = int(tokenizer.max_length / 2)
        func_len_1 = len(func1["input_ids"]) if len(func1["input_ids"]) < truncate_len_1 else truncate_len_1
        truncate_len_2 = tokenizer.max_length - func_len_1
        func_len_2 = len(func2["input_ids"]) if len(func2["input_ids"]) < truncate_len_2 else truncate_len_2
        for keywords in concate_features.keys():
            concate_features[keywords] = func1[keywords][:func_len_1] + func2[keywords][:func_len_2]
        return concate_features

    pair_list = [
        random.sample([func for column, func in example.items() if func and FEATURE_PATTERN.match(column)], 2)
        for example in batch
        if len([func for column, func in example.items() if func and FEATURE_PATTERN.match(column)]) >= 2
    ]

    for idx, pair in enumerate(pair_list):
        # Get the next pair in a circular way, using `zip()` and modulo operator
        next_pair = pair_list[(idx + 1) % len(pair_list)]
        # Concatenate positive pairs
        positive_pairs = [
            concate_pair(pair[0], pair[1]),
            concate_pair(pair[0], pair[1]),
            concate_pair(pair[1], pair[0]),
            concate_pair(pair[1], pair[0]),
        ]
        # Concatenate negative pairs
        negative_pairs = [
            concate_pair(pair[0], next_pair[0]),
            concate_pair(pair[0], next_pair[1]),
            concate_pair(pair[1], next_pair[1]),
            concate_pair(pair[1], next_pair[0]),
        ]
        # Append pairs to feature dictionaries
        for keywords in positive_features:
            positive_features[keywords].extend([positive_pairs[i][keywords] for i in range(4)])
            negative_features[keywords].extend([negative_pairs[i][keywords] for i in range(4)])

    pad_kwargs = dict(
        padding=True, pad_to_multiple_of=8, return_tensors="pt", verbose=False
    )

    batch_positive = tokenizer.pad(positive_features, max_length=max_length, **pad_kwargs)
    batch_negative = tokenizer.pad(negative_features, max_length=max_length, **pad_kwargs)

    return BatchEncoding(batch_positive), BatchEncoding(batch_negative)


def eval_collate_fn(batch):
    def concate_pair(func1, func2):
        concate_features = {f: [] for f in ["input_ids", "attention_mask", "token_type_ids"]}
        truncate_len_1 = int(tokenizer.max_length / 2)
        func_len_1 = len(func1["input_ids"]) if len(func1["input_ids"]) < truncate_len_1 else truncate_len_1
        truncate_len_2 = tokenizer.max_length - func_len_1
        func_len_2 = len(func2["input_ids"]) if len(func2["input_ids"]) < truncate_len_2 else truncate_len_2
        for keywords in concate_features.keys():
            concate_features[keywords] = func1[keywords][:func_len_1] + func2[keywords][:func_len_2]
        return concate_features

    def generate_samples(pair_list, index):
        anchor = pair_list[index][0]
        positive = pair_list[index][1]
        negatives = [p[0] for i, p in enumerate(pair_list) if i != index]
        function_list = [positive] + negatives
        concate_samples = [concate_pair(anchor, func) for func in function_list]
        return concate_samples

    pair_list = [
        random.sample([func for column, func in example.items() if func and FEATURE_PATTERN.match(column)], 2)
        for example in batch
        if len([func for column, func in example.items() if func and FEATURE_PATTERN.match(column)]) >= 2
    ]

    # batch_sample_list shapes like following:
    batch_sample_list = [
        {f: [] for f in ["input_ids", "attention_mask", "token_type_ids"]} for _ in range(len(pair_list))
    ]
    for row in range(len(pair_list)):
        samples = generate_samples(pair_list, row) # [pos_concate, neg_concate_1, neg_concate_2, ..., neg_concate_n]
        for keywords in ["input_ids", "attention_mask", "token_type_ids"]:
            for col, sample in enumerate(samples):
                batch_sample_list[col][keywords].append(sample[keywords])
        
    pad_kwargs = dict(
        padding=True, pad_to_multiple_of=8, return_tensors="pt", verbose=False
    )

    for col in range(len(batch_sample_list)):
        batch_sample_list[col] = BatchEncoding(tokenizer.pad(batch_sample_list[col], **pad_kwargs))

    return batch_sample_list

class CebinTrainer(object):
    def __init__(
        self,
        dataset,
        model,
        tokenizer,
        train_batch_size=1,
        eval_batch_size=1,
        learning_rate=1e-5,
        weight_decay=1e-4,
        margin=0.3,
        train_collator=None,
        eval_collator=None,
        workers=8,
    ):
        """
        Provides an easy to use class for pretraining and evaluating a MoCo Model.
        :param dataset: The dataset to use for training and evaluation.
        :param model: The model to train and evaluate.
        :param tokenizer: The tokenizer to use for encoding the data.
        :param train_batch_size: The batch size to use for training.
        :param eval_batch_size: The batch size to use for evaluation.
        :param learning_rate: The learning rate to use for training.
        :param weight_decay: The weight decay to use for training.
        """
        if train_collator is None or eval_collator is None:
            raise ValueError("train_collator and eval_collator must not be None")

        self.dataset = dataset
        self.model = model
        self.tokenizer = tokenizer
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.train_collator = train_collator
        self.eval_collator = eval_collator
        self.workers = workers
        self.margin = margin

    def build_dataloaders(self, train_shuffle=True, eval_shuffle=True):
        """
        Builds the Training and Eval DataLoaders
        :param train_test_split: The ratio split of test to train data.
        :param train_shuffle: (bool) True if you wish to shuffle the train_dataset.
        :param eval_shuffle: (bool) True if you wish to shuffle the eval_dataset.
        :return: train dataloader and evaluation dataloader.
        """
        train_loader = DataLoader(
            self.dataset["train"],
            collate_fn=self.train_collator,
            batch_size=self.train_batch_size,
            shuffle=train_shuffle,
            num_workers=self.workers
        )

        eval_loader = DataLoader(
            self.dataset["test"],
            collate_fn=self.eval_collator,
            batch_size=self.eval_batch_size,
            shuffle=eval_shuffle,
            num_workers=self.workers
        )

        logger.info(
            f"""train_dataloader size: {len(train_loader.dataset)} | shuffle: {train_shuffle}
                eval_dataloader size: {len(eval_loader.dataset)} | shuffle: {eval_shuffle}"""
        )
        return train_loader, eval_loader

    def train(
        self,
        epochs,
        train_dataloader,
        eval_dataloader,
        log_steps,
        ckpt_steps,
        eval_steps,
        ckpt_dir=None,
        resume_from=None,
        accelerator=None
    ):
        """
        Trains the Reformer Model
        :param epochs: The number of epochs to train for.
        :param train_dataloader: The dataloader to use for training.
        :param eval_dataloader: The dataloader to use for evaluation.
        :param log_steps: The number of steps to log the training loss.
        :param ckpt_steps: The number of steps to save a checkpoint.
        :param eval_steps: The number of steps to evaluate the model.
        :param ckpt_dir: The directory to save the checkpoints.
        :param accelerator: The accelerator to use for training.
        :return: None
        """

        # Set up the optimizer
        logger.info(f"{datetime.now()} | Setting up optimizer...")
        logger.info(
            f"{datetime.now()} | Traning steps: {self.train_batch_size * epochs * len(train_dataloader)}"
        )

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        global_steps = 0
        local_steps = 0
        step_loss = 0.0

        self.model.train()
        get_tflops_func = partial(
            get_tflops, numel, self.train_batch_size, self.tokenizer.max_len
        )

        logger.info(
            f"{datetime.now()} | train_batch_size: {self.train_batch_size} | eval_batch_size: {self.eval_batch_size}"
        )
        logger.info(
            f"{datetime.now()} | Epochs: {epochs} | log_steps: {log_steps} | ckpt_steps: {ckpt_steps} | eval_steps: {eval_steps}"
        )

        (
            self.model,
            optimizer,
            train_dataloader,
            eval_dataloader,
        ) = accelerator.prepare(
            self.model, optimizer, train_dataloader, eval_dataloader
        )

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            int(epochs * len(train_dataloader) * 0.2),
            epochs * len(train_dataloader),
        )
        scheduler = accelerator.prepare(scheduler)

        accelerator.register_for_checkpointing(scheduler)

        if ckpt_dir is not None and resume_from > 0:
            assert os.path.isdir(ckpt_dir)
            try:
                logger.info(f"{datetime.now()} | Continuing from checkpoint...")
                accelerator.load_state(f"{ckpt_dir}/checkpoint-{resume_from}")
                accelerator.skip_first_batches(train_dataloader, resume_from)
                logger.info(f"{datetime.now()} | Resuming from step: {resume_from}")
                global_steps += resume_from
            except Exception as e:
                logger.info(f"{datetime.now()} | No checkpoint was found | {e}")

        local_time = []
        local_tflops = []

        criterion = TripletLoss(margin=0.3)

        for epoch in tqdm(range(epochs), desc="Epochs", position=0):
            logger.info(f"{datetime.now()} | Epoch: {epoch}")
            if resume_from:
                epoch_iterator = tqdm(
                    enumerate(train_dataloader),
                    desc="Epoch Iterator",
                    position=1,
                    leave=True,
                    total=len(train_dataloader),
                    initial=resume_from % len(train_dataloader),
                )
            else:
                epoch_iterator = tqdm(
                    enumerate(train_dataloader),
                    desc="Epoch Iterator",
                    position=1,
                    leave=True,
                    total=len(train_dataloader),
                )

            for step, (positive, negative) in epoch_iterator:
                with accelerator.accumulate(model):
                    start = time()
                    positive_logits = model(**positive)
                    negative_logits = model(**negative)
                    loss = criterion(positive_logits, negative_logits)
                    accelerator.backward(loss)
                    step_loss += loss.item()
                    local_steps += 1
                    global_steps += 1

                    optimizer.step()
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), 1)

                    scheduler.step()
                    optimizer.zero_grad()
                    step_time = time() - start
                    step_tflops = get_tflops_func(step_time)
                    local_time.append(step_time)
                    local_tflops.append(step_tflops)

                if global_steps % log_steps == 0:
                    logger.info(
                        f"{datetime.now()} | Learning Rate: {optimizer.param_groups[0]['lr']} | Step Time: {sum(local_time) / local_steps} | TFLOPs: {sum(local_tflops) / local_steps} | Train Loss: {step_loss / local_steps} | Steps: {global_steps}",
                        main_process_only=True,
                    )
                    accelerator.log(
                        {
                            "Train/Loss": step_loss / local_steps,
                            "Train/Learning Rate": optimizer.param_groups[0]["lr"],
                            "Train/TFLOPs": sum(local_tflops) / local_steps,
                        },
                        step=global_steps,
                    )

                    step_loss = 0.0
                    local_steps = 0
                    local_time = []
                    local_tflops = []

            (
                eval_mrr,
                eval_recall_at_1,
                eval_recall_at_5,
                eval_recall_at_10
            ) = self.evaluate(
                eval_dataloader, accelerator
            )

            accelerator.log(
                {
                    "Eval/MRR": eval_mrr,
                    "Eval/Recall@1": eval_recall_at_1,
                    "Eval/Recall@5": eval_recall_at_5,
                    "Eval/Recall@10": eval_recall_at_10
                },
                step=global_steps,
            )

            # if accelerator.is_main_process and epoch == epochs - 1:
            #     az_logger.log("Eval/MRR", eval_mrr)
            #     az_logger.log("Eval/Recall@1", eval_recall_at_1)
            #     az_logger.log("Eval/Recall@5", eval_recall_at_5)
            #     az_logger.log("Eval/Recall@10", eval_recall_at_10)
            #     az_logger.log("Eval/Recall@50", eval_recall_at_50)
            #     az_logger.log("Eval/Recall@100", eval_recall_at_100)

            accelerator.wait_for_everyone()
            accelerator.save_state(f"{ckpt_dir}/checkpoint-{global_steps}")
            model_to_save = accelerator.unwrap_model(model)
            torch.save(
                model_to_save.state_dict(),
                f"{ckpt_dir}/checkpoint-{global_steps}/CEBin_pair_model_state_dict.pt",
            )
            logger.info(
                f"{datetime.now()} | Saved checkpoint to: {ckpt_dir}/checkpoint-{global_steps}"
            )

        accelerator.wait_for_everyone()
        model_to_save = accelerator.unwrap_model(model)
        torch.save(model_to_save.state_dict(), f"{ckpt_dir}/CEBin_pair_final_state_dict.pt")

    def calculate_mrr_and_recall_at_K(self, logits, labels):
        # logits: Nx(1+K)
        # labels: N
        # mrr: scalar
        # recall: scalar
        # sort the logits in descending order
        _, indices = torch.sort(logits, dim=1, descending=True)
        # get the rank of the positive label for each query
        ranks = torch.nonzero(indices == labels.view(-1, 1), as_tuple=False)[:, 1]
        # convert the rank to reciprocal rank
        mrr = torch.reciprocal(ranks.float() + 1)
        # calculate recall@k
        recall_at_1 = (ranks < 1).float()
        recall_at_5 = (ranks < 5).float()
        recall_at_10 = (ranks < 10).float()

        return (
            mrr.mean(),
            recall_at_1.mean(),
            recall_at_5.mean(),
            recall_at_10.mean()
        )

    def evaluate(self, dataloader, accelerator):
        """
        Runs through the provided dataloader with torch.no_grad()
        :param dataloader: (torch.utils.data.DataLoader) Evaluation DataLoader
        :param accelerator: (Accelerator) Accelerator object
        :return: None
        """

        eval_mrr = 0.0
        eval_recall_at_1 = 0.0
        eval_recall_at_5 = 0.0
        eval_recall_at_10 = 0.0
        eval_steps = 0
        self.model.eval()
        with torch.no_grad():
            for step, batch in tqdm(
                enumerate(dataloader),
                desc="Evaluating",
                leave=True,
                total=len(dataloader),
            ):
                logits_list = []
                for sample in batch:
                    logits_list.append(model(**sample))
                logits = torch.cat(logits_list, dim=1)
                labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)
                (
                    mrr,
                    recall_at_1,
                    recall_at_5,
                    recall_at_10
                ) = self.calculate_mrr_and_recall_at_K(logits, labels)
                
                gather_output = accelerator.gather(
                    [
                        mrr,
                        recall_at_1,
                        recall_at_5,
                        recall_at_10
                    ]
                )

                (
                    tmp_mrr,
                    tmp_recall_at_1,
                    tmp_recall_at_5,
                    tmp_recall_at_10
                ) = gather_output

                eval_mrr += tmp_mrr.mean().item()
                eval_recall_at_1 += tmp_recall_at_1.mean().item()
                eval_recall_at_5 += tmp_recall_at_5.mean().item()
                eval_recall_at_10 += tmp_recall_at_10.mean().item()
                eval_steps += 1

                logger.info(
                    f"{datetime.now()} | Step: {step} | MRR: {mrr} | Recall@1: {recall_at_1} | Recall@5: {recall_at_5} | Recall@10: {recall_at_10}"
                )

            eval_mrr /= eval_steps
            eval_recall_at_1 /= eval_steps
            eval_recall_at_5 /= eval_steps
            eval_recall_at_10 /= eval_steps

            return (
                eval_mrr,
                eval_recall_at_1,
                eval_recall_at_5,
                eval_recall_at_10
            )


def parse_args():
    parser = argparse.ArgumentParser(
        description="PyTorch BinaryNinja Medium Level IL Contrastive Learning"
    )

    parser.add_argument("--data_dir", type=str, default=None, help="The data dir")

    parser.add_argument("--output_dir", type=str, default=None, help="The output dir")

    parser.add_argument(
        "--dataset", type=str, default="BinaryCorp", help="The dataset name"
    )

    parser.add_argument(
        "--pretrain_path", type=str, default="../models/cebin", help="The dataset name"
    )

    parser.add_argument(
        "--tokenizer", type=str, default="cebin-tokenizer", help="The tokneizer name"
    )

    parser.add_argument(
        "--project_name",
        type=str,
        default="cebin-pretrain",
        help="The project name used for wandb logging",
    )

    parser.add_argument(
        "--job_name",
        type=str,
        default="local",
        help="The job name used for wandb logging",
    )

    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="The count for gradient accumulation steps",
    )

    parser.add_argument(
        "--max_len",
        type=int,
        default=1024,
        help="The count for max sequence length",
    )

    parser.add_argument(
        "--train_batchsize",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--eval_batchsize",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--resume_from",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--margin",
        type=float,
        default=0.3,
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--freeze_layers",
        type=int,
        default=22,
    )

    parser.add_argument(
        "--log_steps",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--ckpt_steps",
        type=int,
        default=50000,
    )

    parser.add_argument(
        "--eval_steps",
        type=int,
        default=250000,
    )

    parser.add_argument(
        "--keywords",
        type=str,
        default="all",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    # set_cpu_maximum_parallelism()
    accelerator = Accelerator(
        project_dir=args.output_dir,
        log_with=args.report_to,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    print(args.job_name)
    init_kwargs = None
    if args.job_name is not None:
        init_kwargs = {"wandb": {"name": args.job_name}}

    project_name = args.project_name
    experiment_config = vars(args)
    accelerator.init_trackers(
        project_name,
        experiment_config,
        init_kwargs=init_kwargs,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        # filename=f"{args.output_dir}/log.txt",
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.DEBUG,
    )
    logger.info(accelerator.state, main_process_only=False)

    # datasets.utils.logging.disable_progress_bar()
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    print("loading tokenizers")
    tokenizer = CebinTokenizer.from_pretrained(f"{args.data_dir}/{args.tokenizer}")
    tokenizer.max_length = args.max_len
    tokenizer.max_len = args.max_len

    print("finish loading tokenizer")

    dataset = load_dataset(
        accelerator,
        args.dataset,
        tokenizer,
        use_cache=True,
        cache_dir=args.data_dir,
    )

    model = CEBinPairEncoder(args.pretrain_path)
    for param in model.encoder.embeddings.parameters():
        param.requires_grad = False

    if args.freeze_layers != -1:
        for layer in model.encoder.encoder.layer[:args.freeze_layers]:
            for param in layer.parameters():
                param.requires_grad = False

    numel = get_model_size(model)
    model_size = model_size_formatter(numel)
    logger.info(f"{datetime.now()} | Model size: {model_size}")

    trainer = CebinTrainer(
        dataset,
        model,
        tokenizer,
        train_batch_size=args.train_batchsize,
        eval_batch_size=args.eval_batchsize,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        margin=args.margin,
        train_collator=collate_fn,
        eval_collator=eval_collate_fn,
        workers=args.workers
    )

    train_dataloader, eval_dataloader = trainer.build_dataloaders()

    trainer.train(
        epochs=args.epochs,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        log_steps=args.log_steps,
        ckpt_steps=args.ckpt_steps,
        eval_steps=args.eval_steps,
        ckpt_dir=args.output_dir,
        resume_from=args.resume_from,
        accelerator=accelerator,
    )
