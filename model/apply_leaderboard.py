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
            
            # 1. Randomize Seed
            if "seed_val = 42" in text:
                text = text.replace("seed_val = 42", 'seed_val = random.randint(1, 100000)\nprint(f"🎲 Random Seed: {seed_val}")')
                
            # 2. Add Leaderboard logging at the very end of the training cell
            if "TRAINING_MODE = getattr(CONFIG" in text and "model.predict" in text:
                if "# --- Centralized Leaderboard Logger ---" not in text:
                    logger_code = f"""

# --- Centralized Leaderboard Logger ---
import csv
import os
import shutil
from datetime import datetime

log_file = r'training_leaderboard.csv'
file_exists = os.path.isfile(log_file)
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

old_model_name = f"AI_Brain_v{v}.zip"
new_model_name = f"AI_Brain_v{v}_Seed{{seed_val}}_Profit_{{total_profit:.2f}}.zip"

if os.path.exists(old_model_name):
    shutil.copy(old_model_name, new_model_name)

with open(log_file, mode='a', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(['Timestamp', 'Version', 'Seed', 'Profit (%)', 'Model Path'])
    writer.writerow([timestamp, f"v{v}", seed_val, round(total_profit, 2), new_model_name])
print(f"\\n🎯 Đã lưu log vào Leaderboard! Lợi nhuận: {{total_profit:.2f}}% | Model: {{new_model_name}}")
"""
                    text += logger_code
            
            lines = [line + '\n' for line in text.split('\n')]
            if lines: lines[-1] = lines[-1][:-1]
            cell["source"] = lines

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Injected Leaderboard to v{v}")
