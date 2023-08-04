# %%
import pandas as pd
import numpy as np
from datasets import Dataset, load_from_disk
import json
import os
import os
import torch
import torch.nn as nn
from tokenizer import CebinTokenizer
from transformers import BatchEncoding
from models import RetrivalEncoder
from tqdm import tqdm
import time
import faiss

import os
from tqdm import tqdm

cve_func_path = f"../data/cve-functions.csv"
cve_func_df = pd.read_csv(cve_func_path)
cve_list = os.listdir("./results")

ds_embedding = load_from_disk(os.path.join("./results", cve_list[0], "ds_embedding"))
ds_vul_embedding = load_from_disk(os.path.join("./results", cve_list[0], "vul_embedding"))
# %%
ds_embedding.add_faiss_index(column='embedding', metric_type=faiss.METRIC_INNER_PRODUCT)
func_name = cve_func_df[cve_func_df['cve'] == cve_list[0]]['func_name'].to_list()[0]
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

eval(ds_vul_embedding['embedding'], ds_embedding)
# %%
