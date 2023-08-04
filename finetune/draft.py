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

dataset = load_from_disk("../data/BinaryCorp")
model = CEBinPairEncoder("../data/cebin")
tokenizer = CebinTokenizer.from_pretrained("../data/cebin-tokenizer")
tokenizer.max_length = 1024

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


train_loader = DataLoader(
    dataset["test"],
    collate_fn=collate_fn,
    batch_size=4,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
)

device = torch.device("cuda:0")

# %%
from tqdm import tqdm
with torch.no_grad():
    for batch in tqdm(train_loader):
        positive, negative = batch
        print(len(positive))
        print(len(negative))
        break
# %%
# model = model.to(device)

# %%
from tqdm import tqdm
cnt = 0
with torch.no_grad():
    for batch in tqdm(train_loader):
        scores_list = [

        ]
        for sample in batch:
            sample = sample.to(device)
            sample_score_tensor = model(**sample)
            print("sample tensor: ", sample_score_tensor.shape)
            scores_list.append(sample_score_tensor)
        
        all_score = torch.cat(scores_list, dim=0)
        print(all_score.shape)
        break
        # query = batch[0]
        # print(query)
        # print(query['input_ids'].shape)
        # print(len(batch))
        # if cnt > 100:
        #     break
        # cnt += 1
# %%
all_score = torch.cat(scores_list, dim=1)
print(all_score.shape)
# %%
print(all_score)
# %%
