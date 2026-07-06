import pandas as pd
import re
import json
import os

docs_path = r'c:\Users\ADMIN\Desktop\Kaggle\docs\mapping.txt'
industries_path = r'c:\Users\ADMIN\Desktop\Kaggle\src\data_collection\industries.csv'
out_path = r'c:\Users\ADMIN\Desktop\Kaggle\src\data_collection\icb_mapping.json'

# Parse 46 stocks
with open(docs_path, 'r', encoding='utf-8') as f:
    content = f.read()

symbols = re.findall(r'^-\s+([A-Z0-9]{3}):', content, flags=re.MULTILINE)
if not symbols:
    symbols = re.findall(r'^([A-Z0-9]{3}):\s', content, flags=re.MULTILINE)

symbols = sorted(list(set(symbols)))

# Get ICB level 2
df_icb = pd.read_csv(industries_path)
df_icb2 = df_icb[df_icb['icb_level'] == 2]

mapping = {}
for sym in symbols:
    row = df_icb2[df_icb2['symbol'] == sym]
    if not row.empty:
        mapping[sym] = row.iloc[0]['icb_name']
    else:
        mapping[sym] = 'Unknown'

with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(mapping, f, ensure_ascii=False, indent=4)

print(f"Generated {out_path} with {len(mapping)} symbols.")
for k, v in mapping.items():
    print(f"{k}: {v}")
