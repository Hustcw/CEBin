# %%
import pandas as pd
import numpy as np
from datasets import Dataset
import json
import os
import os
import torch
import torch.nn as nn
from tokenizer import CebinTokenizer
from transformers import BatchEncoding
from models import RetrivalEncoder
from tqdm import tqdm

import faiss

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cve', type=str, default="CVE-2004-0421")
parser.add_argument('--device', type=int, default=0)

args = parser.parse_args()

# cve = "CVE-2004-0421"
cve = args.cve
cve_path = f"../../cve-dataset/{cve}.tsv"
cve_func_path = f"../data/cve-functions.csv"

if os.path.exists(f"./results/{cve}"):
    print(f"Already exists: {cve}")
    exit()

cve_df = pd.read_csv(cve_path, sep="\t")
cve_func_df = pd.read_csv(cve_func_path)

if cve not in cve_func_df['cve'].to_list():
    print(f"{cve} not on the list")
    exit()

print("evaluating: ", cve)
func_name = cve_func_df[cve_func_df['cve'] == cve]['func_name'].to_list()[0]

device = torch.device(f"cuda:{args.device}")
model = torch.load("../models/CEBin-Cisco.bin")
model = model.to(device)
tokenizer = CebinTokenizer.from_pretrained("../cebin-tokenizer")
tokenizer.max_length = 1024

# tmp_df = cve_df[:500]
# vul_df = cve_df[cve_df["func_name"] == func_name]
# cve_df = pd.concat([tmp_df, vul_df])

dataset = Dataset.from_pandas(cve_df)

# %%
def load_json(example):
    example['function'] = json.loads(example['function'].replace("'", '"'))
    return example

dataset = dataset.map(load_json, num_proc=8)
# %%
@torch.no_grad()
def encode_key(batch):
    batch_func = batch['function']
    pad_kwargs = dict(
        padding=True, pad_to_multiple_of=8, return_tensors="pt", verbose=False
    )
    batch_func = BatchEncoding(tokenizer.pad(batch_func, **pad_kwargs))
    batch_func = batch_func.to(device)
    with torch.no_grad():
        # embeddings = model.module.encoder_k(**batch_func).cpu().numpy()
        embeddings = model.encoder_k(**batch_func).cpu().numpy()
    return {'embedding': embeddings}

# %%
ds_embedding = dataset.map(encode_key, batched=True, batch_size=64, remove_columns=['function'])
ds_embedding.add_faiss_index(column='embedding', metric_type=faiss.METRIC_INNER_PRODUCT)

# %%
@torch.no_grad()
def encode_query(batch):
    batch_func = batch['function']
    pad_kwargs = dict(
        padding=True, pad_to_multiple_of=8, return_tensors="pt", verbose=False
    )
    batch_func = BatchEncoding(tokenizer.pad(batch_func, **pad_kwargs))
    batch_func = batch_func.to(device)
    with torch.no_grad():
        # embeddings = model.module.encoder_q(**batch_func).cpu().numpy()
        embeddings = model.encoder_q(**batch_func).cpu().numpy()

    return {'embedding': embeddings}

ds_vul = dataset.filter(lambda x: x['func_name'] == func_name)
ds_vul_embedding = ds_vul.map(encode_query, batched=True, batch_size=4, remove_columns=['function'])

# %%
@torch.no_grad()
def eval(vul_emb, ds_emb):
    recall = 0
    cnt = 0
    vul_len = len(vul_emb)
    for query_vec in tqdm(vul_emb):
        q = np.array(query_vec, dtype=np.float32)
        scores, retrieved_examples = ds_emb.get_nearest_examples('embedding', q, k=vul_len)
        predict = [1 if name == func_name else 0 for name in retrieved_examples['func_name'] ]
        tmp_recall = float(sum(predict)) / len(predict)
        recall += tmp_recall
        cnt += 1
    return recall / cnt

recall = eval(ds_vul_embedding['embedding'], ds_embedding)
print(f"CVE: {cve}, FUNC_NAME: {func_name}, RECALL: {recall}, NUM_FUNC: {len(ds_embedding)}")


# %%
if not os.path.exists(f"./results/{cve}"):
    os.mkdir(f"./results/{cve}")
root_path = f"./results/{cve}"
if not os.path.exists(os.path.join(root_path, "result.json")):
    with open(os.path.join(root_path, "result.json"), 'w') as fp:
        json.dump({"cve": cve, "func_name": func_name, "recall": recall, 'poolsize': len(ds_embedding)}, fp)
ds_vul_embedding.save_to_disk(os.path.join(root_path, "vul_embedding"))
ds_embedding.save_faiss_index('embedding', os.path.join(root_path, 'embedding.faiss'))
ds_embedding.drop_index('embedding')
ds_embedding.save_to_disk(os.path.join(root_path, "ds_embedding"))
