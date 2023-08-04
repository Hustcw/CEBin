# %%
import json
import logging
import pandas as pd
from itertools import chain
from collections import defaultdict
from multiprocessing import Pool

import datasets
from datasets import Dataset, Value, DatasetDict
import transformers

from accelerate import Accelerator
from accelerate.logging import get_logger
from tqdm import tqdm

from tokenizer import CebinTokenizer

logger = get_logger(__name__)
debug = False

def tokenize_fn(example):
    keys = list(example.keys())[2:]
    new_feature = {"package" : example["package"], "func_name": example["func_name"]}
    for k in keys:
        # new_key = f"{k}_tokenized"
        new_key = k
        if example[k]:
            new_feature[new_key] = tokenizer.encode_function(json.loads(example[k]))
        else:
            new_feature[new_key] = None
    return new_feature

def load_dataset(accelerator, dataset_name, tokenizor, use_cache=True, cache_dir=None):
    if use_cache and cache_dir is not None:
        logger.info(f"Loading cached dataset from {cache_dir}")
        return datasets.load_from_disk(f"{cache_dir}/{dataset_name}")
    
    dataset = datasets.load_from_disk(f"../data/{dataset_name}")
    dataset = dataset.filter(filter_fn, num_proc=48)
    # import pdb; pdb.set_trace()

    print("finish loading datasets")

    # tokenize
    # def tokenize_fn(example):
    #     keys = list(example.keys())[2:]
    #     keys = [k for k in example if k not in ["package", "func_name"] and example[k]]
    #     new_feature = defaultdict(list)
    #     new_feature["available_keys"] = keys
    #     for k in keys:
    #         func = json.loads(example[k])
    #         rslt = tokenizor.encode_function(func)
    #         for k in rslt:
    #             new_feature[k].append(rslt[k])

    #     return new_feature
    
    # to_remove_features = list(dataset["train"].features)[2:]
                
    # with accelerator.main_process_first():
    #     # tokenized_datasets = dataset.map(
    #     tokenized_datasets = dataset.map(
    #         tokenize_fn,
    #         batched=False,
    #         num_proc=48,
    #         desc=f"tokenize func str",
    #         remove_columns=to_remove_features,
    #         drop_last_batch=True
    #     )
    results_train = []
    results_test = []
    with Pool(processes=96) as pool: # Change the number of processes as per your requirements
        results_test = pool.map(tokenize_fn, dataset["test"])
        results_train = pool.map(tokenize_fn, dataset["train"])
        # for result in tqdm(pool.imap_unordered(tokenize_fn, dataset["test"]), total=len(dataset["test"])):
        #     results_test.append(result)

        # for result in tqdm(pool.imap_unordered(tokenize_fn, dataset["train"]), total=len(dataset["train"])):
        #     results_train.append(result)

    
    train_dataset = Dataset.from_list(results_train)
    test_dataset = Dataset.from_list(results_test)
    binarycorp = DatasetDict({"train": train_dataset, "test": test_dataset})

    # rename_mapping = {}
    # for ftr in list(dataset.features):
    #     if ftr.endswith("tokenized"):
    #         rename_mapping[ftr] = "_".join(ftr.split('_')[:-1])
    
    # tokenized_datasets.rename_columns(rename_mapping)

    if not use_cache and cache_dir is not None:
        binarycorp.save_to_disk(f"{cache_dir}/{dataset_name}")
    return binarycorp

def filter_fn(example):
    cnt = 0
    for k, v in example.items():
        if k != "package" and k != "func_name" and v is not None:
            cnt += 1
    return cnt > 1

if __name__ == "__main__":
    if debug:
        dataset = datasets.load_from_disk(f"../data/BinaryCorp-New")
        dataset = dataset.filter(filter_fn, num_proc=48)
        tokenizer = CebinTokenizer.from_pretrained("../cebin-tokenizer")
        tokenizer.max_length = 4096
        def tokenize_fn(example):
            keys = list(example.keys())[2:]
            new_feature = {}
            for k in keys:
                new_key = f"{k}_tokenized"
                if example[k]:
                    new_feature[new_key] = tokenizer.encode_function(json.loads(example[k]))
                else:
                    new_feature[new_key] = None
            return new_feature
        with Pool(processes=48) as pool: # Change the number of processes as per your requirements
            results = pool.map(tokenize_fn, dataset["test"])
        
    else:
        accelerator = Accelerator()
        max_len = 4096
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
        tokenizer.max_length = max_len
        tokenizer.max_len = max_len
        print("finish loading tokenizer")
        # load_dataset(accelerator, ["Trex"], tokenizor)
        dataset = load_dataset(accelerator, "BinaryCorp", tokenizer, use_cache=False, cache_dir="../cache/BinaryCorp")
# # %%
# def find_invalid_outputs(result):
#     invalid_indices = []
#     for i, d in enumerate(result):
#         for k, v in d.items():
#             if v and len(v) != 0:
#                 pass
#             else:
#                 invalid_indices.append(i)
#                 break
#     return invalid_indices

# # %%
# invalid_indices = find_invalid_outputs(results)
# print(invalid_indices)

# # %%
# for i, d in enumerate(results):
#     for k, v in d.items():
#         print(k, v)
#         print(k, "input_ids" in v)
#         print(type(v))
#         break
#     break
# # %%
# results[0]
# # %%
# t = Dataset.from_list(results)
# # %%
# print(t[0])
# # %%
# print(t[0].keys())
# # %%
# t.features
# # %%
