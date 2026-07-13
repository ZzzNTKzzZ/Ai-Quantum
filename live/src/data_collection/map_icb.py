import pandas as pd
import re

# Read mapping.txt to get the 46 symbols
with open(r'c:\Users\ADMIN\Desktop\Kaggle\docs\mapping.txt', 'r', encoding='utf-8') as f:
    content = f.read()

symbols = re.findall(r'^-\s+([A-Z0-9]{3}):', content, flags=re.MULTILINE)
if not symbols:
    symbols = re.findall(r'^([A-Z0-9]{3}):\s', content, flags=re.MULTILINE)
symbols = list(set(symbols))

df_icb = pd.read_csv(r'c:\Users\ADMIN\Desktop\Kaggle\src\data_collection\industries.csv')
# filter level 2 or 3
df_icb_level = df_icb[df_icb['icb_level'] == 2]

mapping = []
for sym in symbols:
    row = df_icb_level[df_icb_level['symbol'] == sym]
    if not row.empty:
        mapping.append({'symbol': sym, 'icb_name': row.iloc[0]['icb_name']})

df_map = pd.DataFrame(mapping)
print(df_map.groupby('icb_name').size())
