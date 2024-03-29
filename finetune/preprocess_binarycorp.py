import pandas as pd
from datasets import Dataset, DatasetDict
import os
import json

def process_dataframe(df):
    grouped = df.groupby(["package", "func_name"])
    column_combinations = df.apply(lambda x: f"{x['arch']}_{x['compiler']}_{x['optimizer']}", axis=1).unique()
    new_df = pd.DataFrame(columns=["package", "func_name"] + list(column_combinations))

    for group_name, group in grouped:
        package, func_name = group_name
        row = {"package": package, "func_name": func_name}
        for _, entry in group.iterrows():
            column = f"{entry['arch']}_{entry['compiler']}_{entry['optimizer']}"
            row[column] = json.loads(entry["func_str"])
        new_df = new_df.append(row, ignore_index=True)

    return new_df

def fileter_dataframe(df):
    df = df[
        (df["instr_cnt"] >= 10)
        & (df["bb_cnt"] >= 5)
        & (df["instr_cnt"] <= 1024)
        & (df["bb_cnt"] <= 256)
    ]
    return df

train_csv = "/path/to/Cebin/Cebin/data/processed/BinaryCorp/train.tsv"
test_csv = "/path/to/Cebin/Cebin/data/processed/BinaryCorp/test.tsv"
train_df = pd.read_csv(train_csv, sep="\t")
test_df = pd.read_csv(test_csv, sep="\t")

train_df = fileter_dataframe(train_df)
test_df = fileter_dataframe(test_df)

train_processed = process_dataframe(train_df)
test_processed = process_dataframe(test_df)

train_dataset = Dataset.from_pandas(train_processed)
test_dataset = Dataset.from_pandas(test_processed)

binarycorp = DatasetDict({"train": train_dataset, "test": test_dataset})

output_directory = "../data/BinaryCorp"
os.makedirs(output_directory, exist_ok=True)
binarycorp.save_to_disk(output_directory)
