# %%
import re
import random
import torch
from models import CEBinPairEncoder
from torch.utils.data import DataLoader
from tokenizer import CebinTokenizer
from datasets import load_from_disk
from transformers import BatchEncoding

FEATURE_PATTERN = re.compile(r"^(x86|x64|arm|mips)")

dataset = load_from_disk("../data/BinaryCorp")
model = torch.load("../models/CEBin-Pair-Cisco.bin")
tokenizer = CebinTokenizer.from_pretrained("../data/cebin-tokenizer")
tokenizer.max_length = 1024

# %%
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
        positive_pair = concate_pair(pair[0], pair[1])
        
        # Concatenate negative pairs
        negative_pair = concate_pair(pair[0], next_pair[0])
        
        # Append pairs to feature dictionaries
        for keywords in positive_features:
            positive_features[keywords].append(positive_pair[keywords])
            negative_features[keywords].append(negative_pair[keywords])

    pad_kwargs = dict(
        padding=True, pad_to_multiple_of=8, return_tensors="pt", verbose=False
    )

    batch_positive = tokenizer.pad(positive_features, max_length=max_length, **pad_kwargs)
    batch_negative = tokenizer.pad(negative_features, max_length=max_length, **pad_kwargs)

    return BatchEncoding(batch_positive), BatchEncoding(batch_negative)


train_loader = DataLoader(
    dataset["test"],
    collate_fn=collate_fn,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

device = torch.device("cuda:0")
model = model.to(device)

# %%
with torch.no_grad():
    for batch in train_loader:
        positive_pair, negative_pair = batch
        positive_pair, negative_pair = positive_pair.to(device), negative_pair.to(device)
        positive_logits = model(**positive_pair) # batch_size x 1
        negative_logits = model(**negative_pair) # batch_size x 1
        print(positive_logits)
        print(negative_logits)
        break
# %%
