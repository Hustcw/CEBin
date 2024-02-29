# %%
import os
import json
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt

result_dir = "results" 
result_data = defaultdict(list)
for root, dirs, files in os.walk(result_dir):
    for filename in files:
        if filename == 'result.json':
            filepath = os.path.join(root, filename)
            with open(filepath) as f:
                data = json.load(f)
                print(data)
                for key, value in data.items():
                    if key == 'pooslize':
                        key = 'poolsize'
                    result_data[key].append(value)
df = pd.DataFrame(result_data)
print(df)
df.to_csv("results_collection.csv", index=False)
df = df.sort_values(by="poolsize")
plt.plot(df["poolsize"], df["recall"])
plt.xlabel("poolsize")
plt.ylabel("recall")
plt.show()
# %%
