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
from models import CEBinEncoder, MoCo

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

def pick_features(keywords, features):
    if keywords == "all":
        return None
    return set(
        [feature for feature in features if any(key in feature for key in keywords)]
    )


def pool_collate_base_fn(batch, filter=None, max_length=1024):
    func_features = {f: [] for f in ["input_ids", "attention_mask", "token_type_ids"]}
    max_func_len = 0

    def filter_helper(column):
        if filter is None:
            return FEATURE_PATTERN.match(column)
        return column in filter

    for example in batch:
        funcs = [
            func
            for column, func in example.items()
            if filter_helper(column) and func is not None
        ]

        if len(funcs) == 0:
            continue

        func = random.choice(funcs)
        max_func_len = max(max_func_len, len(func["input_ids"][:max_length]))
        for keywords in func_features.keys():
            func_features[keywords].append(func[keywords][:max_length])

    pad_kwargs = dict(
        padding=True, pad_to_multiple_of=8, return_tensors="pt", verbose=False
    )

    batch_function = tokenizer.pad(func_features, max_length=max_func_len, **pad_kwargs)
    return BatchEncoding(batch_function)


def data_collate_fn(batch):
    query_features = {f: [] for f in ["input_ids", "attention_mask", "token_type_ids"]}
    key_features = {f: [] for f in ["input_ids", "attention_mask", "token_type_ids"]}
    max_query_len = max_key_len = 0

    for example in batch:
        cans = [
            func
            for column, func in example.items()
            if func is not None and FEATURE_PATTERN.match(column)
        ]
        # if func is not None and ( column.startswith("x86") or column.startswith("x64") or column.startswith("arm") or column.startswith("mips") )
        if len(cans) < 2:
            continue
        pair = random.sample(cans, 2)

        max_query_len = max(max_query_len, len(pair[0]["input_ids"][: args.max_len]))
        max_key_len = max(max_key_len, len(pair[1]["input_ids"][: args.max_len]))

        for keywords in key_features.keys():
            query_features[keywords].append(pair[0][keywords][: args.max_len])
            key_features[keywords].append(pair[1][keywords][: args.max_len])

    pad_kwargs = dict(
        padding=True, pad_to_multiple_of=8, return_tensors="pt", verbose=False
    )

    batch_query = tokenizer.pad(query_features, max_length=max_query_len, **pad_kwargs)
    batch_key = tokenizer.pad(key_features, max_length=max_key_len, **pad_kwargs)

    return BatchEncoding(batch_query), BatchEncoding(batch_key)


class MoCoTrainer(object):
    def __init__(
        self,
        dataset,
        model,
        tokenizer,
        train_batch_size=1,
        eval_batch_size=1,
        learning_rate=1e-5,
        weight_decay=1e-4,
        data_collator=None,
        pool_collator=None,
        workers=4,
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
        if data_collator is None or pool_collator is None:
            raise ValueError("data_collator and pool_collator must not be None")

        self.dataset = dataset
        self.model = model
        self.tokenizer = tokenizer
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.data_collator = data_collator
        self.pool_collator = pool_collator
        self.workers = workers

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
            collate_fn=self.data_collator,
            batch_size=self.train_batch_size,
            shuffle=train_shuffle,
            num_workers=self.workers,
            pin_memory=True,
        )

        eval_split_dataset = self.dataset["test"].train_test_split(0.05)

        eval_loader = DataLoader(
            eval_split_dataset["test"],
            collate_fn=self.data_collator,
            batch_size=self.eval_batch_size,
            shuffle=eval_shuffle,
            num_workers=self.workers,
            pin_memory=True,
        )

        pool_dataloader = DataLoader(
            eval_split_dataset["train"],
            collate_fn=self.pool_collator,
            batch_size=self.eval_batch_size,
            shuffle=eval_shuffle,
            num_workers=self.workers,
            pin_memory=True,
        )

        logger.info(
            f"""train_dataloader size: {len(train_loader.dataset)} | shuffle: {train_shuffle}
                eval_dataloader size: {len(eval_loader.dataset)} | shuffle: {eval_shuffle}
                pool_dataloader size: {len(pool_dataloader.dataset)} | shuffle: {eval_shuffle}"""
        )
        return train_loader, eval_loader, pool_dataloader

    def train(
        self,
        epochs,
        train_dataloader,
        eval_dataloader,
        pool_dataloader,
        log_steps,
        ckpt_steps,
        eval_steps,
        pool_ratio=1,
        ckpt_dir=None,
        resume_from=None,
        accelerator=None,
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
            pool_dataloader,
        ) = accelerator.prepare(
            self.model, optimizer, train_dataloader, eval_dataloader, pool_dataloader
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

        criterion = nn.CrossEntropyLoss()

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
            for step, (q, k) in epoch_iterator:
                with accelerator.accumulate(model):
                    start = time()
                    logits, target = self.model(q, k)
                    loss = criterion(logits, target)
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
                eval_loss,
                eval_mrr,
                eval_recall_at_1,
                eval_recall_at_5,
                eval_recall_at_10,
                eval_recall_at_50,
                eval_recall_at_100,
            ) = self.evaluate(
                eval_dataloader, pool_dataloader, accelerator, pool_ratio=pool_ratio
            )

            accelerator.log(
                {
                    "Eval/Loss": eval_loss,
                    "Eval/MRR": eval_mrr,
                    "Eval/Recall@1": eval_recall_at_1,
                    "Eval/Recall@5": eval_recall_at_5,
                    "Eval/Recall@10": eval_recall_at_10,
                    "Eval/Recall@50": eval_recall_at_50,
                    "Eval/Recall@100": eval_recall_at_100,
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
                f"{ckpt_dir}/checkpoint-{global_steps}/CEBin_model_state_dict.pt",
            )
            logger.info(
                f"{datetime.now()} | Saved checkpoint to: {ckpt_dir}/checkpoint-{global_steps}"
            )

        accelerator.wait_for_everyone()
        model_to_save = accelerator.unwrap_model(model)
        torch.save(model_to_save.state_dict(), f"{ckpt_dir}/CEBin_final_state_dict.pt")

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
        recall_at_50 = (ranks < 50).float()
        recall_at_100 = (ranks < 100).float()

        return (
            mrr.mean(),
            recall_at_1.mean(),
            recall_at_5.mean(),
            recall_at_10.mean(),
            recall_at_50.mean(),
            recall_at_100.mean(),
        )

    def evaluate(self, dataloader, poolloader, accelerator, pool_ratio=1):
        """
        Runs through the provided dataloader with torch.no_grad()
        :param dataloader: (torch.utils.data.DataLoader) Evaluation DataLoader
        :param accelerator: (Accelerator) Accelerator object
        :return: None
        """

        eval_loss = 0.0
        eval_mrr = 0.0
        eval_recall_at_1 = 0.0
        eval_recall_at_5 = 0.0
        eval_recall_at_10 = 0.0
        eval_recall_at_50 = 0.0
        eval_recall_at_100 = 0.0
        eval_steps = 0

        self.model.eval()
        logger.info(f"{datetime.now()} | Preparing Function Pools...")
        with torch.no_grad():
            function_pool = []
            for _ in range(pool_ratio):
                for step, batch in tqdm(
                    enumerate(poolloader),
                    desc="Function Pools Preparing",
                    leave=True,
                    total=len(poolloader),
                ):
                    # Calculate the function embeddings for the current batch
                    function_embeddings = self.model.encoder_k(**batch)

                    # Append the function embeddings to the function pool
                    function_pool.append(function_embeddings)

            # Concatenate all the function embeddings into a large tensor
            function_pool_tensor = torch.cat(function_pool, dim=0)

            # gather from different device
            function_pool_tensor = accelerator.gather(function_pool_tensor)

            logger.info(f"{datetime.now()} | Pool Tensor: {function_pool_tensor.shape}")

            logger.info(f"{datetime.now()} | Evaluating...")
            criterion = nn.CrossEntropyLoss()
            for _ in range(pool_ratio):
                for step, batch in tqdm(
                    enumerate(dataloader),
                    desc="Evaluating",
                    leave=True,
                    total=len(dataloader),
                ):
                    func_q, func_k = batch
                    q = self.model.encoder_q(**func_q)
                    k = self.model.encoder_k(**func_k)

                    l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
                    l_neg = torch.einsum("nc,ck->nk", [q, function_pool_tensor.T])
                    logits = torch.cat([l_pos, l_neg], dim=1)
                    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)
                    loss = criterion(logits / self.model.T, labels)
                    (
                        mrr,
                        recall_at_1,
                        recall_at_5,
                        recall_at_10,
                        recall_at_50,
                        recall_at_100,
                    ) = self.calculate_mrr_and_recall_at_K(logits, labels)

                    gather_output = accelerator.gather(
                        [
                            loss,
                            mrr,
                            recall_at_1,
                            recall_at_5,
                            recall_at_10,
                            recall_at_50,
                            recall_at_100,
                        ]
                    )

                    (
                        tmp_eval_loss,
                        tmp_mrr,
                        tmp_recall_at_1,
                        tmp_recall_at_5,
                        tmp_recall_at_10,
                        tmp_recall_at_50,
                        tmp_recall_at_100,
                    ) = gather_output

                    eval_loss += tmp_eval_loss.mean().item()
                    eval_mrr += tmp_mrr.mean().item()
                    eval_recall_at_1 += tmp_recall_at_1.mean().item()
                    eval_recall_at_5 += tmp_recall_at_5.mean().item()
                    eval_recall_at_10 += tmp_recall_at_10.mean().item()
                    eval_recall_at_50 += tmp_recall_at_50.mean().item()
                    eval_recall_at_100 += tmp_recall_at_100.mean().item()
                    eval_steps += 1

                    logger.info(
                        f"{datetime.now()} | Step: {step} | Loss: {loss} | MRR: {mrr} | Recall@1: {recall_at_1} | Recall@5: {recall_at_5} | Recall@10: {recall_at_10} | Recall@50: {recall_at_50} | Recall@100: {recall_at_100}"
                    )

            eval_loss /= eval_steps
            eval_mrr /= eval_steps
            eval_recall_at_1 /= eval_steps
            eval_recall_at_5 /= eval_steps
            eval_recall_at_10 /= eval_steps
            eval_recall_at_50 /= eval_steps
            eval_recall_at_100 /= eval_steps

            return (
                eval_loss,
                eval_mrr,
                eval_recall_at_1,
                eval_recall_at_5,
                eval_recall_at_10,
                eval_recall_at_50,
                eval_recall_at_100,
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
        default=4,
        help="The count for gradient accumulation steps",
    )

    parser.add_argument(
        "--max_len",
        type=int,
        default=4096,
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
        "--moco_t",
        type=float,
        default=0.07,
    )

    parser.add_argument(
        "--moco_m",
        type=float,
        default=0.99,
    )

    parser.add_argument(
        "--moco_dim",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--moco_k",
        type=int,
        default=65536,
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
        "--pool_ratio",
        type=int,
        default=3,
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

    model = MoCo(
        CEBinEncoder,
        accelerator=accelerator,
        pretrain_path=args.pretrain_path,
        freeze_layers=args.freeze_layers,
        dim=args.moco_dim,
        K=args.moco_k,
        m=args.moco_m,
        T=args.moco_t,
    )
    numel = get_model_size(model)
    model_size = model_size_formatter(numel)
    logger.info(f"{datetime.now()} | Model size: {model_size}")

    trainer = MoCoTrainer(
        dataset,
        model,
        tokenizer,
        train_batch_size=args.train_batchsize,
        eval_batch_size=args.eval_batchsize,
        learning_rate=args.learning_rate,
        data_collator=data_collate_fn,
        pool_collator=partial(
            pool_collate_base_fn,
            filter=pick_features(args.keywords, DATASET_FEATURES[args.dataset]),
            max_length=args.max_len,
        ),
    )

    train_dataloader, eval_dataloader, pool_dataloader = trainer.build_dataloaders()

    trainer.train(
        epochs=args.epochs,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        pool_dataloader=pool_dataloader,
        log_steps=args.log_steps,
        ckpt_steps=args.ckpt_steps,
        eval_steps=args.eval_steps,
        pool_ratio=args.pool_ratio,
        ckpt_dir=args.output_dir,
        resume_from=args.resume_from,
        accelerator=accelerator,
    )
