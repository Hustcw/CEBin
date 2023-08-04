# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
from tokenizer import CebinTokenizer
import random
from models import CEBinEncoder, MoCo
from transformers import BatchEncoding
from tqdm import tqdm
from accelerate import Accelerator

accelerator = Accelerator()

tokenizer = CebinTokenizer.from_pretrained("../cebin-tokenizer")

model = MoCo(CEBinEncoder, accelerator=accelerator, freeze_layers=20)
# %%
dataset = load_from_disk("../cache/BinaryCorp")
batch_size = 2
# Modify the collate_fn function

# %%
# def collate_fn(batch):
#     batch_input_ids = []
#     batch_attention_masks = []
#     batch_type_ids = []
#     batch_labels = []

#     max_length = 0
#     for idx, example in enumerate(batch):
#         for k, v in example.items():
#             if k.startswith("x86") and v is not None:
#                 batch_input_ids.append(v["input_ids"])
#                 batch_attention_masks.append(v["attention_mask"])
#                 batch_type_ids.append(v["token_type_ids"])
#                 batch_labels.append(idx)
#                 max_length = max(max_length, len(v["input_ids"]))

#     features = {
#         "input_ids": batch_input_ids,
#         "attention_mask": batch_attention_masks,
#         "token_type_ids": batch_type_ids,
#         "labels": batch_labels
#     }
#     print(max_length)

#     return tokenizer.pad(
#         features, 
#         padding=True,
#         max_length=max_length,
#         pad_to_multiple_of=8,
#         return_tensors="pt"
#     )

def collate_fn(batch):
    query_features = {f: [] for f in ["input_ids", "attention_mask", "token_type_ids"]}
    key_features = {f: [] for f in ["input_ids", "attention_mask", "token_type_ids"]}
    max_query_len = max_key_len = 0

    for example in batch:
        cans = [func for column, func in example.items() if column.startswith("x86") and func is not None]
        pair = random.sample(cans, 2)
        
        max_query_len = max(max_query_len, len(pair[0]["input_ids"]))
        max_key_len = max(max_key_len, len(pair[1]["input_ids"]))

        for keywords in key_features.keys():
            query_features[keywords].append(pair[0][keywords])
            key_features[keywords].append(pair[1][keywords])

    pad_kwargs = dict(
        padding=True,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )

    batch_query = tokenizer.pad(query_features, max_length=max_query_len, **pad_kwargs)
    batch_key = tokenizer.pad(key_features, max_length=max_key_len, **pad_kwargs)
    
    return BatchEncoding(batch_query), BatchEncoding(batch_key)

test_loader = DataLoader(
    dataset['test'],
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
)
# %%
device = torch.device("cuda:0")
model = model.to(device)
counter = 0
# logits_list = []
# %%
example = next(iter(test_loader))

# %%
func_q, func_k = example
func_q, func_k = func_q.to(device), func_k.to(device)
q, k = model.encoder_q(**func_q), model.encoder_k(**func_k)
l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
l_neg = torch.einsum("nc,ck->nk", [q, model.queue.clone().detach()])
logits = torch.cat([l_pos, l_neg], dim=1)
labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
_, indices = torch.sort(logits, dim=1, descending=True)
# %%
ranks = torch.nonzero(indices == labels.view(-1, 1), as_tuple=False)[:, 1]
print(ranks)
mrr = torch.reciprocal(ranks.float() + 1)
recall = (ranks < 1).float()
# %%
def evaluate(model, func_q, func_k):
    with torch.no_grad():  # no gradient to keys
        # compute query features
        q = model.encoder_q(**func_q)  # queries: NxC
        k = model.encoder_k(**func_k)  # keys: NxC

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, model.queue.clone().detach()])
        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # calculate mrr and recall@k for the batch using logits and labels
        mrr, recall = calculate_mrr_and_recall(logits, labels)


def calculate_mrr_and_recall(logits, labels, K):
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
    recall = (ranks < K).float()

    return mrr.mean().item(), recall.mean().item()

for sample in tqdm(test_loader):
    q, k = sample
    q, k = q.to(device), k.to(device)
    logits, labels = model(q, k)
    # logits_list.append(logits.cpu().detach().numpy())
    # logits_list.append(logits)
    # print(q['input_ids'].shape)
    # print(k['token_type_ids'].shape)
    if counter > 4:
        break
    counter += 1

# %%
# logits_list = [logits.squeeze().tolist() for logits in logits_list]

# %% 
for logits in logits_list:
    print(logits)
# %%
dataset_raw = load_from_disk("../data/BinaryCorp")

def filter_fn(example):
    cnt = 0
    for k, v in example.items():
        if k.startswith("x86") and v is not None:
            cnt += 1
    return cnt > 1

filter_dataset_raw = dataset_raw.filter(filter_fn, num_proc=48)

def count_data(ds):
    cnt = 0
    features = list(ds.features)[2:]
    for k in features:
        cnt += len(ds.filter(lambda x: x[k] == None, num_proc=48))
    return len(ds) * len(features) - cnt

# %%
