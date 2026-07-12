import json

with open(r"c:\Users\ADMIN\Desktop\Kaggle\model\ppo_grouped_rl_v8.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell["cell_type"] == "code":
        text = "".join(cell["source"])
        if "model.predict" in text:
            with open("predict_block.txt", "w", encoding="utf-8") as out:
                out.write(text)
