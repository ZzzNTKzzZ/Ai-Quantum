import vnstock
import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')

ref = vnstock.Reference()
df = ref.equity.list_by_industry()
if isinstance(df, pd.DataFrame):
    df.to_csv('industries.csv', index=False, encoding='utf-8-sig')
else:
    pd.DataFrame(df).to_csv('industries.csv', index=False, encoding='utf-8-sig')
