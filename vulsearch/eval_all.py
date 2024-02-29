# %%
import subprocess
import pandas as pd
import os
from tqdm import tqdm

cve_func_path = f"../data/cve-functions.csv"
cve_func_df = pd.read_csv(cve_func_path)
cve_set = set(cve_func_df['cve'].to_list())
cve_list_file = os.listdir("../../cve-dataset")
cve_list = [cve_file.split('.')[0] for cve_file in cve_list_file if cve_file.split('.')[0] in cve_set]
skip_list = os.listdir("./results")
cve_list = [cve for cve in cve_list if cve not in skip_list]
device_list = [0, 1, 2, 3]

devices_in_use = {}

running_tasks = []

for cve in tqdm(cve_list):
    while len(devices_in_use) == len(device_list):
        for device, task in devices_in_use.items():
            if task.poll() is not None:
                del devices_in_use[device]
                break

    for device in device_list:
        if device not in devices_in_use:
            cmd = f"python3 ann-builder.py --cve {cve} --device {device}"
            task = subprocess.Popen(cmd, shell=True)
            devices_in_use[device] = task
            running_tasks.append(task)
            break

for task in running_tasks:
    task.wait()
