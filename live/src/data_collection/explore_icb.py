import vnstock
import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')

ref = vnstock.Reference()
df = ref.equity.list()
df.to_csv('all_stocks_icb.csv', index=False, encoding='utf-8-sig')
print(df.columns)
if 'industry' in df.columns:
    print(df['industry'].unique())
elif 'icb_code' in df.columns:
    print(df['icb_code'].unique())
elif 'icb_name' in df.columns:
    print(df['icb_name'].unique())
else:
    print("Columns are:", df.columns)
