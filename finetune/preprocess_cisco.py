import pandas as pd
from datasets import Dataset, DatasetDict
import os
import json
import numpy as np
import swifter
from tqdm.auto import tqdm
from datetime import datetime
from tokenizer import CebinTokenizer
from sklearn.model_selection import train_test_split

arch = {'x86', 'arm32', 'mips64', 'mips32', 'arm64', 'x64'}
compiler = {'gcc4.8', 'clang3.5', 'clang5.0', 'clang9', 'gcc5', 'gcc7', 'gcc9', 'clang7'}
opt = {'O3', 'O0', 'O1', 'Os', 'O2'}

def fileter_dataframe(df):
    df = df[
        (df["instr_cnt"] >= 30)
        & (df["bb_cnt"] >= 5)
        & (df["instr_cnt"] <= 1024)
        & (df["bb_cnt"] <= 256)
    ]
    return df

# 读取 train.csv 和 test.csv
test_csv = "/mnt/data_fast/vul337/Cebin/Cebin/data/processed/Cisco/test.tsv"
train_csv = "/mnt/data_fast/vul337/Cebin/Cebin/data/processed/Cisco/train.tsv"

train_df = pd.read_csv(train_csv, sep="\t")
test_df = pd.read_csv(test_csv, sep="\t")
# 过滤掉不符合要求的样本
merge_df = pd.concat([train_df, test_df], ignore_index=True)
merge_df = fileter_dataframe(merge_df)
merge_df = merge_df.groupby(['package', 'func_name']).filter(lambda x: len(x) > 1)
merge_df = pd.concat([group for _, group in merge_df.groupby(['package', 'func_name'])], ignore_index=True)

tokenizer = CebinTokenizer.from_pretrained("../cebin-tokenizer")
tokenizer.max_length = 4096
tokenizer.max_len = 4096
print("finish loading tokenizer")

def estimate_remaining_time(start_time, processed, total):
    start_time_dt = datetime.fromtimestamp(start_time)
    elapsed_time = (datetime.now() - start_time_dt).total_seconds()
    remaining_time = (elapsed_time / processed) * (total - processed)
    return str(pd.to_timedelta(remaining_time, unit='s'))

def process_group(group, progress_bar):
    row = {'_'.join([a, c, o]):None for a in arch for c in compiler for o in opt}
    for _, entry in group.iterrows():
        try:
            column = f"{entry['arch']}_{entry['compiler']}_{entry['optimizer']}"
        except:
            continue
        row[column] = tokenizer.encode_function(json.loads(entry["func_str"]))
    
    progress_bar.update(1)
    progress_bar.set_postfix(remaining_time=estimate_remaining_time(progress_bar.start_t, progress_bar.n, progress_bar.total))
    return row

def convert(new_df):
    rows = ['_'.join([a, c, o]) for a in arch for c in compiler for o in opt]
    new_df = new_df.to_frame(name="val").reset_index()
    for row in tqdm(rows):
        new_df[row] = new_df.apply(lambda x: x["val"][row] if row in x["val"] else None, axis=1)
    new_df = new_df.drop(columns=["val"])
    return new_df

def process_dataframe(df):
    print(f"{datetime.now()}: Processing dataframe with {len(df)} rows")
    grouped = df.groupby(["package", "func_name"])
    print(f"{datetime.now()}: Grouped dataframe into {len(grouped)} groups")

    with tqdm(total=len(grouped), desc="Processing groups") as progress_bar:
        new_df = grouped.apply(lambda group: process_group(group, progress_bar))
        
    return convert(new_df)

from multiprocessing import Pool, cpu_count
workers = 96
df_parts = np.array_split(merge_df, workers)
with Pool(workers) as p:
    processed_dfs = p.map(process_dataframe, df_parts)
new_df = pd.concat(processed_dfs, ignore_index=True)
rows = ['_'.join([a, c, o]) for a in arch for c in compiler for o in opt]
new_df = new_df[new_df[rows].notnull().sum(axis=1) > 1]

unique_packages = new_df["package"].unique()
train_packages, test_packages = train_test_split(unique_packages, test_size=0.2, random_state=42)
train_df = new_df[new_df["package"].isin(train_packages)]
test_df = new_df[new_df["package"].isin(test_packages)]

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

Cisco = DatasetDict({"train": train_dataset, "test": test_dataset})
Cisco.save_to_disk("../cache/Cisco/")