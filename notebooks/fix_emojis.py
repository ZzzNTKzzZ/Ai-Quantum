import json

path = r'c:\Users\ADMIN\Desktop\Kaggle\notebooks\f.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source_str = "".join(cell['source'])
        if "Các đặc trưng ĐƯỢC GIỮ LẠI" in source_str:
            new_source = source_str.replace("✅", "[KEEP]").replace("❌", "[DROP]")
            cell['source'] = [line + '\n' for line in new_source.split('\n')]
            cell['source'] = [line.replace('\n\n', '\n') for line in cell['source']]
            break

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Removed emojis from cell to fix UnicodeEncodeError.")
