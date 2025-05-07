import re
import random
import torch
from models import RetrivalEncoder
from torch.utils.data import DataLoader
from tokenizer import CebinTokenizer
from datasets import load_from_disk
from transformers import BatchEncoding

FEATURE_PATTERN = re.compile(r"^(x86|x64|arm|mips)")
DATASET_FEATURES = {
    "BinaryCorp": [
        "x86-64_gcc-11_O1",
        "x86-64_gcc-11_Os",
        "x86-64_gcc-11_O3",
        "x86-64_gcc-11_O0",
        "x86-64_gcc-11_O2",
    ],
    "Trex": [
        "arm32_gcc7.5_O3",
        "arm32_gcc7.5_O2",
        "arm32_gcc7.5_O1",
        "arm32_gcc7.5_Os",
        "arm32_gcc7.5_O0",
        "mips32_gcc7.5_O3",
        "mips32_gcc7.5_O2",
        "mips32_gcc7.5_O1",
        "mips32_gcc7.5_Os",
        "mips32_gcc7.5_O0",
        "arm64_gcc7.5_O3",
        "arm64_gcc7.5_O2",
        "arm64_gcc7.5_O1",
        "arm64_gcc7.5_Os",
        "arm64_gcc7.5_O0",
        "x64_gcc7.5_O3",
        "x64_gcc7.5_O2",
        "x64_gcc7.5_O1",
        "x64_gcc7.5_Os",
        "x64_gcc7.5_O0",
        "x86_gcc7.5_O3",
        "x86_gcc7.5_O2",
        "x86_gcc7.5_O1",
        "x86_gcc7.5_Os",
        "x86_gcc7.5_O0",
        "mips64_gcc7.5_O3",
        "mips64_gcc7.5_O2",
        "mips64_gcc7.5_O1",
        "mips64_gcc7.5_Os",
        "mips64_gcc7.5_O0",
    ],
    "Cisco": [
        "x64_clang3.5_O2",
        "x64_clang3.5_O3",
        "x64_clang3.5_O0",
        "x64_clang3.5_O1",
        "x64_clang3.5_Os",
        "x64_clang7_O2",
        "x64_clang7_O3",
        "x64_clang7_O0",
        "x64_clang7_O1",
        "x64_clang7_Os",
        "x64_gcc4.8_O2",
        "x64_gcc4.8_O3",
        "x64_gcc4.8_O0",
        "x64_gcc4.8_O1",
        "x64_gcc4.8_Os",
        "x64_clang9_O2",
        "x64_clang9_O3",
        "x64_clang9_O0",
        "x64_clang9_O1",
        "x64_clang9_Os",
        "x64_clang5.0_O2",
        "x64_clang5.0_O3",
        "x64_clang5.0_O0",
        "x64_clang5.0_O1",
        "x64_clang5.0_Os",
        "x64_gcc7_O2",
        "x64_gcc7_O3",
        "x64_gcc7_O0",
        "x64_gcc7_O1",
        "x64_gcc7_Os",
        "x64_gcc5_O2",
        "x64_gcc5_O3",
        "x64_gcc5_O0",
        "x64_gcc5_O1",
        "x64_gcc5_Os",
        "x64_gcc9_O2",
        "x64_gcc9_O3",
        "x64_gcc9_O0",
        "x64_gcc9_O1",
        "x64_gcc9_Os",
        "mips32_clang3.5_O2",
        "mips32_clang3.5_O3",
        "mips32_clang3.5_O0",
        "mips32_clang3.5_O1",
        "mips32_clang3.5_Os",
        "mips32_clang7_O2",
        "mips32_clang7_O3",
        "mips32_clang7_O0",
        "mips32_clang7_O1",
        "mips32_clang7_Os",
        "mips32_gcc4.8_O2",
        "mips32_gcc4.8_O3",
        "mips32_gcc4.8_O0",
        "mips32_gcc4.8_O1",
        "mips32_gcc4.8_Os",
        "mips32_clang9_O2",
        "mips32_clang9_O3",
        "mips32_clang9_O0",
        "mips32_clang9_O1",
        "mips32_clang9_Os",
        "mips32_clang5.0_O2",
        "mips32_clang5.0_O3",
        "mips32_clang5.0_O0",
        "mips32_clang5.0_O1",
        "mips32_clang5.0_Os",
        "mips32_gcc7_O2",
        "mips32_gcc7_O3",
        "mips32_gcc7_O0",
        "mips32_gcc7_O1",
        "mips32_gcc7_Os",
        "mips32_gcc5_O2",
        "mips32_gcc5_O3",
        "mips32_gcc5_O0",
        "mips32_gcc5_O1",
        "mips32_gcc5_Os",
        "mips32_gcc9_O2",
        "mips32_gcc9_O3",
        "mips32_gcc9_O0",
        "mips32_gcc9_O1",
        "mips32_gcc9_Os",
        "arm64_clang3.5_O2",
        "arm64_clang3.5_O3",
        "arm64_clang3.5_O0",
        "arm64_clang3.5_O1",
        "arm64_clang3.5_Os",
        "arm64_clang7_O2",
        "arm64_clang7_O3",
        "arm64_clang7_O0",
        "arm64_clang7_O1",
        "arm64_clang7_Os",
        "arm64_gcc4.8_O2",
        "arm64_gcc4.8_O3",
        "arm64_gcc4.8_O0",
        "arm64_gcc4.8_O1",
        "arm64_gcc4.8_Os",
        "arm64_clang9_O2",
        "arm64_clang9_O3",
        "arm64_clang9_O0",
        "arm64_clang9_O1",
        "arm64_clang9_Os",
        "arm64_clang5.0_O2",
        "arm64_clang5.0_O3",
        "arm64_clang5.0_O0",
        "arm64_clang5.0_O1",
        "arm64_clang5.0_Os",
        "arm64_gcc7_O2",
        "arm64_gcc7_O3",
        "arm64_gcc7_O0",
        "arm64_gcc7_O1",
        "arm64_gcc7_Os",
        "arm64_gcc5_O2",
        "arm64_gcc5_O3",
        "arm64_gcc5_O0",
        "arm64_gcc5_O1",
        "arm64_gcc5_Os",
        "arm64_gcc9_O2",
        "arm64_gcc9_O3",
        "arm64_gcc9_O0",
        "arm64_gcc9_O1",
        "arm64_gcc9_Os",
        "mips64_clang3.5_O2",
        "mips64_clang3.5_O3",
        "mips64_clang3.5_O0",
        "mips64_clang3.5_O1",
        "mips64_clang3.5_Os",
        "mips64_clang7_O2",
        "mips64_clang7_O3",
        "mips64_clang7_O0",
        "mips64_clang7_O1",
        "mips64_clang7_Os",
        "mips64_gcc4.8_O2",
        "mips64_gcc4.8_O3",
        "mips64_gcc4.8_O0",
        "mips64_gcc4.8_O1",
        "mips64_gcc4.8_Os",
        "mips64_clang9_O2",
        "mips64_clang9_O3",
        "mips64_clang9_O0",
        "mips64_clang9_O1",
        "mips64_clang9_Os",
        "mips64_clang5.0_O2",
        "mips64_clang5.0_O3",
        "mips64_clang5.0_O0",
        "mips64_clang5.0_O1",
        "mips64_clang5.0_Os",
        "mips64_gcc7_O2",
        "mips64_gcc7_O3",
        "mips64_gcc7_O0",
        "mips64_gcc7_O1",
        "mips64_gcc7_Os",
        "mips64_gcc5_O2",
        "mips64_gcc5_O3",
        "mips64_gcc5_O0",
        "mips64_gcc5_O1",
        "mips64_gcc5_Os",
        "mips64_gcc9_O2",
        "mips64_gcc9_O3",
        "mips64_gcc9_O0",
        "mips64_gcc9_O1",
        "mips64_gcc9_Os",
        "arm32_clang3.5_O2",
        "arm32_clang3.5_O3",
        "arm32_clang3.5_O0",
        "arm32_clang3.5_O1",
        "arm32_clang3.5_Os",
        "arm32_clang7_O2",
        "arm32_clang7_O3",
        "arm32_clang7_O0",
        "arm32_clang7_O1",
        "arm32_clang7_Os",
        "arm32_gcc4.8_O2",
        "arm32_gcc4.8_O3",
        "arm32_gcc4.8_O0",
        "arm32_gcc4.8_O1",
        "arm32_gcc4.8_Os",
        "arm32_clang9_O2",
        "arm32_clang9_O3",
        "arm32_clang9_O0",
        "arm32_clang9_O1",
        "arm32_clang9_Os",
        "arm32_clang5.0_O2",
        "arm32_clang5.0_O3",
        "arm32_clang5.0_O0",
        "arm32_clang5.0_O1",
        "arm32_clang5.0_Os",
        "arm32_gcc7_O2",
        "arm32_gcc7_O3",
        "arm32_gcc7_O0",
        "arm32_gcc7_O1",
        "arm32_gcc7_Os",
        "arm32_gcc5_O2",
        "arm32_gcc5_O3",
        "arm32_gcc5_O0",
        "arm32_gcc5_O1",
        "arm32_gcc5_Os",
        "arm32_gcc9_O2",
        "arm32_gcc9_O3",
        "arm32_gcc9_O0",
        "arm32_gcc9_O1",
        "arm32_gcc9_Os",
        "x86_clang3.5_O2",
        "x86_clang3.5_O3",
        "x86_clang3.5_O0",
        "x86_clang3.5_O1",
        "x86_clang3.5_Os",
        "x86_clang7_O2",
        "x86_clang7_O3",
        "x86_clang7_O0",
        "x86_clang7_O1",
        "x86_clang7_Os",
        "x86_gcc4.8_O2",
        "x86_gcc4.8_O3",
        "x86_gcc4.8_O0",
        "x86_gcc4.8_O1",
        "x86_gcc4.8_Os",
        "x86_clang9_O2",
        "x86_clang9_O3",
        "x86_clang9_O0",
        "x86_clang9_O1",
        "x86_clang9_Os",
        "x86_clang5.0_O2",
        "x86_clang5.0_O3",
        "x86_clang5.0_O0",
        "x86_clang5.0_O1",
        "x86_clang5.0_Os",
        "x86_gcc7_O2",
        "x86_gcc7_O3",
        "x86_gcc7_O0",
        "x86_gcc7_O1",
        "x86_gcc7_Os",
        "x86_gcc5_O2",
        "x86_gcc5_O3",
        "x86_gcc5_O0",
        "x86_gcc5_O1",
        "x86_gcc5_Os",
        "x86_gcc9_O2",
        "x86_gcc9_O3",
        "x86_gcc9_O0",
        "x86_gcc9_O1",
        "x86_gcc9_Os",
    ],
}

dataset = load_from_disk("../dataset/BinaryCorp")
model = torch.load("../models/CEBin-Embedding-Cisco.bin")
tokenizer = CebinTokenizer.from_pretrained("../data/cebin-tokenizer")
tokenizer.max_length = 1024


def collate_fn(batch):
    query_features = {f: [] for f in ["input_ids", "attention_mask", "token_type_ids"]}
    key_features = {f: [] for f in ["input_ids", "attention_mask", "token_type_ids"]}
    max_query_len = max_key_len = 0

    for example in batch:
        cans = [
            func
            for column, func in example.items()
            if func is not None and FEATURE_PATTERN.match(column)
        ]
        if len(cans) < 2:
            continue
        pair = random.sample(cans, 2)

        max_query_len = max(
            max_query_len, len(pair[0]["input_ids"][: tokenizer.max_length])
        )
        max_key_len = max(
            max_key_len, len(pair[1]["input_ids"][: tokenizer.max_length])
        )

        for keywords in key_features.keys():
            query_features[keywords].append(pair[0][keywords][: tokenizer.max_length])
            key_features[keywords].append(pair[1][keywords][: tokenizer.max_length])

    pad_kwargs = dict(
        padding=True, pad_to_multiple_of=8, return_tensors="pt", verbose=False
    )

    batch_query = tokenizer.pad(query_features, max_length=max_query_len, **pad_kwargs)
    batch_key = tokenizer.pad(key_features, max_length=max_key_len, **pad_kwargs)

    return BatchEncoding(batch_query), BatchEncoding(batch_key)


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
        query, key = batch
        query, key = query.to(device), key.to(device)
        V_q = model.encoder_q(**query)  # batch_size x dim
        V_k = model.encoder_k(**key)  # batch_size x dim
        print(V_q.shape, V_k.shape)
        logits = torch.einsum("nc,nc->n", [V_q, V_k]).unsqueeze(-1)
        print(logits.shape)
        print(logits)
        break
