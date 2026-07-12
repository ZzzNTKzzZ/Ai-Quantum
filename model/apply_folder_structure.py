import json
import re

versions = [5, 6, 7, 8]

for v in versions:
    filepath = rf"c:\Users\ADMIN\Desktop\Kaggle\model\ppo_grouped_rl_v{v}.ipynb"
    with open(filepath, "r", encoding="utf-8") as f:
        nb = json.load(f)
        
    for cell in nb.get("cells", []):
        if cell["cell_type"] == "code":
            text = "".join(cell["source"])
            
            # Create folder at the beginning where the seed is
            if "import os\n" in text and "os.makedirs" not in text:
                text = text.replace("import os\n", f"import os\nos.makedirs('v{v}', exist_ok=True)\n")
                
            # Update VecNormalize
            text = text.replace('"vec_normalize.pkl"', f'"v{v}/vec_normalize.pkl"')
            text = text.replace("'vec_normalize.pkl'", f'"v{v}/vec_normalize.pkl"')
            
            # Update model save paths
            text = text.replace(f'"AI_Brain_v{v}.zip"', f'"v{v}/AI_Brain_v{v}.zip"')
            text = text.replace(f"'AI_Brain_v{v}.zip'", f'"v{v}/AI_Brain_v{v}.zip"')
            
            # Update leaderboard paths
            text = text.replace(f'old_model_name = f"AI_Brain_v{v}.zip"', f'old_model_name = f"v{v}/AI_Brain_v{v}.zip"')
            text = text.replace(f'new_model_name = f"AI_Brain_v{v}_Seed{{seed_val}}_Profit_{{total_profit:.2f}}.zip"', f'new_model_name = f"v{v}/AI_Brain_v{v}_Seed{{seed_val}}_Profit_{{total_profit:.2f}}.zip"')

            lines = [line + '\n' for line in text.split('\n')]
            if lines: lines[-1] = lines[-1][:-1]
            cell["source"] = lines

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Updated paths for v{v}")
