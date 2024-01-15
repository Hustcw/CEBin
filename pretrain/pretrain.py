import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

from tqdm import tqdm

import transformers
from transformers import (
    BertTokenizer,
    PreTrainedTokenizer,
    DataCollatorWithPadding,
    DataCollatorForLanguageModeling,
)
from transformers import Adafactor, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import OneCycleLR
from transformers import RoFormerConfig
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
from models import JRoFormerForMaskedLM

from helper import (
    get_tflops,
    model_size_formatter,
    set_cpu_maximum_parallelism,
    get_model_size,
)
from functools import partial
from time import time
import argparse

logger = get_logger(__name__)


def get_one_cycle(optimizer, num_training_steps):
    """Simple single-cycle scheduler. Not including paper/fastai three-phase things or asymmetry."""

    def lr_lambda(current_step):
        if current_step < num_training_steps / 2:
            return float(current_step / (num_training_steps / 2))
        else:
            return float(2 - current_step / (num_training_steps / 2))

    return LambdaLR(optimizer, lr_lambda, -1)


class JDataCollatorForLanguageModeling(DataCollatorForLanguageModeling):
    def __init__(
        self,
        tokenizer,
        mlm=True,
        mlm_probability=0.15,
        pad_to_multiple_of=None,
        tf_experimental_compile=False,
        return_tensors="pt",
    ):
        super().__init__(
            tokenizer,
            mlm,
            mlm_probability,
            pad_to_multiple_of,
            tf_experimental_compile,
            return_tensors,
        )

    def torch_mask_tokens(self, inputs, special_tokens_mask):
        """
        Prepare masked tokens inputs/labels for masked language modeling
        """
        import torch

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True
                )
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 1.0)).bool() & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        return inputs, labels


class ReformerTrainer(object):
    def __init__(
        self,
        dataset,
        model,
        tokenizer,
        device=None,
        train_batch_size=32,
        eval_batch_size=None,
        mlm_probability=0.15,
        pad_to_multiple_of=8,
        learning_rate=2e-4,
        weight_decay=0.01,
        betas=(0.9, 0.98),
        epsilon=1e-12,
        tb_writer=False,
    ):
        """
        Provides an easy to use class for pretraining and evaluating a Reformer Model.
        :param dataset: (torch.utils.data.Dataset) containing all of the data you wish to utilize during training.
        :param model: (reformer_pytorch.Reformer)
        :param tokenizer: (transformers.PreTrainedTokenizer) defaults to BertTokenizer ('bert-base-case')
        """

        self.dataset = dataset
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.tb_writer = tb_writer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.betas = betas
        self.epsilon = epsilon
        self.data_collator = JDataCollatorForLanguageModeling(
            self.tokenizer,
            mlm_probability=mlm_probability,
            pad_to_multiple_of=pad_to_multiple_of,
        )

        if tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

        if device is None:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        if eval_batch_size is None:
            self.eval_batch_size = train_batch_size

    def build_dataloaders(
        self, train_test_split=0.1, train_shuffle=True, eval_shuffle=True
    ):
        """
        Builds the Training and Eval DataLoaders
        :param train_test_split: The ratio split of test to train data.
        :param train_shuffle: (bool) True if you wish to shuffle the train_dataset.
        :param eval_shuffle: (bool) True if you wish to shuffle the eval_dataset.
        :return: train dataloader and evaluation dataloader.
        """
        dataset_len = len(self.dataset)
        eval_len = int(dataset_len * train_test_split)
        train_len = dataset_len - eval_len
        train_dataset, eval_dataset = random_split(self.dataset, (train_len, eval_len))
        train_loader = DataLoader(
            train_dataset,
            collate_fn=self.data_collator,
            batch_size=self.train_batch_size,
            shuffle=train_shuffle,
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.eval_batch_size,
            collate_fn=self.data_collator,
            shuffle=eval_shuffle,
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
        accelerator=None,
    ):
        """
        Trains the Reformer Model
        :param epochs: The number of times you wish to loop through the dataset.
        :param train_dataloader: (torch.utils.data.DataLoader) The data to train on.
        :param eval_dataloader: (torch.utils.data.DataLoader) The data to evaluate on.
        :param log_steps: The number of steps to iterate before logging.
        :param ckpt_steps: The number of steps to iterate before checkpointing.
        :param ckpt_dir: The directory to save the checkpoints to.
        :param gradient_accumulation_steps: Optional gradient accumulation.
        :return: Total number of steps, total loss, model
        """

        # Set up the optimizer
        logger.info(f"{datetime.now()} | Setting up optimizer...")
        logger.info(
            f"{datetime.now()} | Traning steps: {self.train_batch_size * epochs * len(train_dataloader) // accelerator.gradient_accumulation_steps}"
        )

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            betas=self.betas,
            eps=self.epsilon,
            weight_decay=self.weight_decay,
        )

        # scheduler = OneCycleLR(
        #     optimizer, max_lr=1e-3, epochs=epochs, steps_per_epoch=len(train_dataloader)
        # )
        # scheduler = get_linear_schedule_with_warmup(optimizer, self.num_warmup_steps, self.num_training_steps)
        # optimizer = Adafactor(self.model.parameters())
        # loss_fn = nn.CrossEntropyLoss()
        losses = {}
        global_steps = 0
        local_steps = 0
        step_loss = 0.0

        if ckpt_dir is not None:
            assert os.path.isdir(ckpt_dir)
            try:
                logger.info(f"{datetime.now()} | Continuing from checkpoint...")
                self.model.load_state_dict(
                    torch.load(f"{ckpt_dir}/model_state_dict.pt", map_location="cpu")
                )
                optimizer.load_state_dict(
                    torch.load(f"{ckpt_dir}/optimizer_state_dict.pt")
                )

            except Exception as e:
                logger.info(f"{datetime.now()} | No checkpoint was found | {e}")

        self.model.eval()
        get_tflops_func = partial(
            get_tflops, numel, self.train_batch_size, self.tokenizer.max_len
        )

        # if self.n_gpu > 1:
        #     self.model = nn.DataParallel(self.model)
        #     logging.info(f"{datetime.now()} | Utilizing {self.n_gpu} GPUs")

        logger.info(
            f"{datetime.now()} | train_batch_size: {self.train_batch_size} | eval_batch_size: {self.eval_batch_size}"
        )
        logger.info(
            f"{datetime.now()} | Epochs: {epochs} | log_steps: {log_steps} | ckpt_steps: {ckpt_steps} | eval_steps: {eval_steps}"
        )
        # logging.info(
        #     f"{datetime.now()} | gradient_accumulation_steps: {gradient_accumulation_steps}"
        # )

        (
            self.model,
            optimizer,
            train_dataloader,
            eval_dataloader,
        ) = accelerator.prepare(
            self.model, optimizer, train_dataloader, eval_dataloader
        )

        # scheduler = get_one_cycle(optimizer, epochs * len(train_dataloader))
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            int(epochs * len(train_dataloader) * 0.2),
            epochs * len(train_dataloader),
        )
        scheduler = accelerator.prepare(scheduler)

        accelerator.register_for_checkpointing(scheduler)

        local_time = []
        local_tflops = []
        for epoch in tqdm(range(epochs), desc="Epochs", position=0):
            logger.info(f"{datetime.now()} | Epoch: {epoch}")
            for step, batch in tqdm(
                enumerate(train_dataloader),
                desc="Epoch Iterator",
                position=1,
                leave=True,
                total=len(train_dataloader),
            ):
                with accelerator.accumulate(model):
                    # batch["input_ids"], batch["labels"] = self.mask_tokens(
                    #     batch["input_ids"]
                    # )
                    # batch = batch
                    start = time()
                    output = self.model(**batch)
                    loss = output.loss

                    # if gradient_accumulation_steps > 1:
                    #     loss /= gradient_accumulation_steps
                    # import pdb; pdb.set_trace()
                    # loss.backward()
                    accelerator.backward(loss)
                    # bwd_end = time()
                    # bwd_time = bwd_end - fwd_end
                    step_loss += loss.item()
                    # losses[global_steps] = loss.item()
                    local_steps += 1
                    global_steps += 1

                    optimizer.step()
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), 1)

                    scheduler.step()
                    optimizer.zero_grad()
                    # self.model.zero_grad()

                    step_time = time() - start
                    step_tflops = get_tflops_func(step_time)
                    # logger.info(
                    #     f"{datetime.now()} | Step time: {step_time:.3f}s, TFLOPS: {step_tflops:.3f}, FWD time: {fwd_time:.3f}s, BWD time: {bwd_time:.3f}s",
                    # )
                    local_time.append(step_time)
                    local_tflops.append(step_tflops)

                if global_steps % log_steps == 0:
                    # if self.tb_writer:
                    #     self.writer.add_scalar(
                    #         "Train/Loss", step_loss / local_steps, global_steps
                    #     )
                    #     self.writer.close()

                    # logging learning rate

                    logger.info(
                        f"{datetime.now()} | Learning Rate: {optimizer.param_groups[0]['lr']} | Step Time: {sum(local_time) / local_steps} | TFLOPs: {sum(local_tflops) / local_steps} | Train Loss: {step_loss / local_steps} | Steps: {global_steps}",
                        main_process_only=False,
                    )
                    accelerator.log(
                        {
                            "Train/Loss": step_loss / local_steps,
                            "Train/Learning Rate": optimizer.param_groups[0]["lr"],
                            "TFLOPs": sum(local_tflops) / local_steps,
                        },
                        step=global_steps,
                    )

                    step_loss = 0.0
                    local_steps = 0
                    local_time = []
                    local_tflops = []

                if global_steps % eval_steps == 0:
                    eval_loss, ppl = self.evaluate(eval_dataloader, accelerator)
                    accelerator.log(
                        {"Eval/Loss": eval_loss, "Eval/Perplexity": ppl},
                        step=global_steps,
                    )

                if global_steps % ckpt_steps == 0:
                    # evaluating before every checkpoint
                    accelerator.wait_for_everyone()
                    # model_to_save = accelerator.unwrap_model(model)
                    # accelerator.save(
                    #     model_to_save.state_dict(), f"{ckpt_dir}/model_state_dict.pt"
                    # )
                    # accelerator.save(
                    #     optimizer.state_dict(), f"{ckpt_dir}/optimizer_state_dict.pt"
                    # )
                    accelerator.save_state(f"{ckpt_dir}/checkpoint-{global_steps}")
                    model_to_save = accelerator.unwrap_model(model)
                    model_to_save.save_pretrained(
                        f"{ckpt_dir}/checkpoint-{global_steps}/cebin-large",
                        is_main_process=accelerator.is_main_process,
                        save_function=accelerator.save,
                    )
                    # model_to_save = (
                    #     self.model.module
                    #     if hasattr(self.model, "module")
                    #     else self.model
                    # )

                    # torch.save(
                    #     model_to_save.state_dict(), f"{ckpt_dir}/model_state_dict.pt"
                    # )

                    # torch.save(
                    #     optimizer.state_dict(), f"{ckpt_dir}/optimizer_state_dict.pt"
                    # )
                    logger.info(
                        f"{datetime.now()} | Saved checkpoint to: {ckpt_dir}/checkpoint-{global_steps}"
                    )
        accelerator.wait_for_everyone()
        model_to_save = accelerator.unwrap_model(model)
        model_to_save.save_pretrained(
            f"{ckpt_dir}/cebin-large",
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )
        return self.model

    def evaluate(self, dataloader, accelerator):
        """
        Runs through the provided dataloader with torch.no_grad()
        :param dataloader: (torch.utils.data.DataLoader) Evaluation DataLoader
        :return: None
        """
        # loss_fn = nn.CrossEntropyLoss()

        # if self.n_gpu > 1 and not isinstance(self.model, nn.DataParallel):
        #     self.model = nn.DataParallel(self.model)

        self.model.eval()
        eval_loss = 0.0
        perplexity = 0.0
        eval_steps = 0

        # dataloader = accelerator.prepare(eval_dataloader)

        logger.info(f"{datetime.now()} | Evaluating...")
        for step, batch in tqdm(
            enumerate(dataloader), desc="Evaluating", leave=True, total=len(dataloader)
        ):
            # batch["input_ids"], batch["labels"] = self.mask_tokens(batch["input_ids"])
            with torch.no_grad():
                output = self.model(**batch)

            all_output = accelerator.gather([output.loss])
            # import pdb; pdb.set_trace()
            # loss_mx = labels != -100
            # output_ids = output[loss_mx].view(-1, self.tokenizer.vocab_size)
            # labels = labels[loss_mx].view(-1)
            tmp_eval_loss = all_output[0].mean()
            tmp_perplexity = torch.exp(tmp_eval_loss)

            # if self.n_gpu > 1:
            #     tmp_eval_loss = tmp_eval_loss.mean()

            eval_loss += tmp_eval_loss.item()
            perplexity += tmp_perplexity.item()
            eval_steps += 1

        eval_loss /= eval_steps
        perplexity /= eval_steps

        logger.info(
            f"{datetime.now()} | Step: {step} | Eval Loss: {eval_loss} | Perplexity: {perplexity}"
        )

        return eval_loss, perplexity


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pretrain a JRoformer model on BinaryNinja Medium Level IL"
    )

    parser.add_argument("--data_dir", type=str, default=None, help="The data dir")

    parser.add_argument("--output_dir", type=str, default=None, help="The output dir")

    parser.add_argument(
        "--data_name", type=str, default="Cisco_BinaryCorp_Trex", help="The data dir"
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
        default=8,
        help="The count for gradient accumulation steps",
    )

    parser.add_argument(
        "--max_len",
        type=int,
        default=512,
        help="The count for max sequence length",
    )

    parser.add_argument(
        "--train_batchsize",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--eval_batchsize",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--log_steps",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--ckpt_steps",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--eval_steps",
        type=int,
        default=150000,
    )

    parser.add_argument(
        "--train_test_split",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=1e-2, 
        help="Weight decay to use."
    )

    args = parser.parse_args()

    return args


print("test")
if __name__ == "__main__":
    args = parse_args()
    print("parsed args")
    set_cpu_maximum_parallelism()
    accelerator = Accelerator(
        project_dir=args.output_dir,
        log_with=args.report_to,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

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

    # dataset = load_dataset(accelerator, ["Cisco", "BinaryCorp", "Trex"], tokenizer, use_cache=False, save_cache=True, cache_dir="/path/to/cebin-data/cache/Cisco_BinaryCorp_Trex")
    dataset = load_dataset(
        args.data_dir,
        accelerator,
        ["Cisco", "BinaryCorp", "Trex"],
        tokenizer,
        use_cache=True,
        save_cache=False,
        cache_dir=f"{args.data_dir}/{args.data_name}",
    )
    # dataset = load_dataset(accelerator, ["Cisco", "BinaryCorp", "Trex"], tokenizer, use_cache=True, save_cache=False, cache_dir="../data/cache/trex")
    # dataset = load_dataset(accelerator, ["Cisco", "BinaryCorp", "Trex"], tokenizer, use_cache=True, save_cache=False, cache_dir="../data/cache/trex")

    config = RoFormerConfig(
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=8192,
        num_hidden_layers=24,
        num_attention_heads=16,
        hidden_size=1024,
        pad_token_id=1,
        rotary_value=False,
        use_bias=False,
    )

    # if accelerator.is_main_process:
    #     config.save_pretrained(f"{args.output_dir}")

    model = JRoFormerForMaskedLM(config)
    numel = get_model_size(model)
    model_size = model_size_formatter(numel)
    logger.info(f"{datetime.now()} | Model size: {model_size}")

    trainer = ReformerTrainer(
        dataset,
        model,
        tokenizer,
        train_batch_size=args.train_batchsize,
        eval_batch_size=args.eval_batchsize,
        weight_decay=args.weight_decay
    )

    train_dataloader, eval_dataloader = trainer.build_dataloaders(
        train_test_split=args.train_test_split
    )

    model = trainer.train(
        epochs=args.epochs,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        log_steps=args.log_steps,
        ckpt_steps=args.ckpt_steps,
        eval_steps=args.eval_steps,
        ckpt_dir=args.output_dir,
        accelerator=accelerator,
    )
