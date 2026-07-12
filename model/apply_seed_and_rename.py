import json

versions = [5, 6, 7, 8]

for v in versions:
    filepath = rf"c:\Users\ADMIN\Desktop\Kaggle\model\ppo_grouped_rl_v{v}.ipynb"
    with open(filepath, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    for cell in nb.get("cells", []):
        if cell["cell_type"] == "code":
            text = "".join(cell["source"])
            
            # 1. Add Seed if not present
            if "import torch as th" in text and "seed_val = 42" not in text:
                seed_code = """
import random
seed_val = 42
th.manual_seed(seed_val)
np.random.seed(seed_val)
random.seed(seed_val)
# Thiết lập thêm cho tính ổn định của môi trường Gym
import os
os.environ['PYTHONHASHSEED'] = str(seed_val)
"""
                text = text.replace("import torch as th\n", "import torch as th\n" + seed_code)
                
            # 2. Rename model save path
            if 'AI_Brain_Current.zip' in text:
                text = text.replace('AI_Brain_Current.zip', f'AI_Brain_v{v}.zip')
                
            lines = [line + '\n' for line in text.split('\n')]
            if lines: lines[-1] = lines[-1][:-1]
            cell["source"] = lines

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Updated v{v}")
