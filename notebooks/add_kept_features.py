import json

path = r'c:\Users\ADMIN\Desktop\Kaggle\notebooks\f.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_str = "".join(cell['source'])
        if "selected_raw_features = stat_df[stat_df['is_stationary']]['feature'].tolist()" in source_str and "ĐƯỢC GIỮ LẠI" not in source_str:
            add_lines = [
                "\nprint(f'\\n✅ Các đặc trưng ĐƯỢC GIỮ LẠI ({len(selected_raw_features)} biến): {selected_raw_features}')\n",
                "dropped = [c for c in daily_pool if c not in selected_raw_features]\n",
                "print(f'❌ Các đặc trưng BỊ LOẠI BỎ ({len(dropped)} biến): {dropped}')\n"
            ]
            cell['source'].extend(add_lines)
            break

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Added summary print statements successfully.")
