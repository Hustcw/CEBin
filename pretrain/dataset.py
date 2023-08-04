import json
import logging
import pandas as pd
from itertools import chain

import datasets
from datasets import Dataset
import transformers

from accelerate import Accelerator
from accelerate.logging import get_logger

from tokenizer import CebinTokenizer

logger = get_logger(__name__)


def load_dataset(
    dataset_path,
    accelerator,
    data_name_list,
    tokenizor,
    use_cache=True,
    save_cache=False,
    cache_dir=None,
):
    if use_cache and cache_dir is not None:
        print(f"Loading cached dataset from {cache_dir}")
        logger.info(f"Loading cached dataset from {cache_dir}")
        return Dataset.load_from_disk(cache_dir)

    df_list = []
    for data_name in data_name_list:
        df = pd.read_csv(
            f"{dataset_path}/{data_name}/train.tsv",
            sep="\t",
            usecols=["func_str", "instr_cnt", "bb_cnt"],
        )
        df_list.append(df)
    df = pd.concat(df_list)
    # import pdb; pdb.set_trace()
    # df = pd.read_csv(
    #     f"../data/processed/{data_name}/test.tsv",
    #     sep="\t",
    # )
    logger.info("finish loading tsv")

    # filter out the functions with less than 10 instructions and less than 5 bb_cnt
    df = df[
        (df["instr_cnt"] >= 10)
        & (df["bb_cnt"] >= 5)
        & (df["instr_cnt"] <= 1024)
        & (df["bb_cnt"] <= 256)
    ]

    logger.info("finish filtering tsv")

    df = df[["func_str"]]
    df.reset_index(drop=True, inplace=True)

    logger.info("finish reset index")

    dataset = Dataset.from_pandas(df)

    logger.info("finish converting to dataset")

    # tokenize
    def tokenize_fn(example):
        func = json.loads(example["func_str"])
        rslt = tokenizor.encode_function(func)
        return rslt

    with accelerator.main_process_first():
        tokenized_datasets = dataset.map(
            tokenize_fn,
            batched=False,
            num_proc=48,
            remove_columns=["func_str"],
            desc=f"tokenize func str",
        )

    max_seq_length = 512

    def group_fn(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= max_seq_length:
            total_length = (total_length // max_seq_length + 1) * max_seq_length
        # Split by chunks of max_len.
        result = {
            k: [
                t[i : i + max_seq_length]
                for i in range(0, total_length, max_seq_length)
            ]
            for k, t in concatenated_examples.items()
        }
        return result

    with accelerator.main_process_first():
        process_datasets = tokenized_datasets.map(
            group_fn,
            batched=True,
            batch_size=1,
            num_proc=24,
            desc=f"Splitting texts in chunks of {max_seq_length}",
        )

    if save_cache and cache_dir is not None:
        process_datasets.save_to_disk(cache_dir)
    return process_datasets


if __name__ == "__main__":
    accelerator = Accelerator(gradient_accumulation_steps=256)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    tokenizer = CebinTokenizer.from_pretrained("../cebin-tokenizer")
    tokenizer.max_length = 512
    tokenizer.max_len = 512
    print("finish loading tokenizer")
    # load_dataset(accelerator, ["Trex"], tokenizor)
    dataset = load_dataset(
        accelerator,
        ["Cisco", "BinaryCorp", "Trex"],
        tokenizer,
        use_cache=False,
        save_cache=True,
        cache_dir="/home/vul337/cebin-data/cache/Cisco_BinaryCorp_Trex",
    )
