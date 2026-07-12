import json
import re

with open(r"c:\Users\ADMIN\Desktop\Kaggle\model\ppo_grouped_rl_v6.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell["cell_type"] == "code":
        text = "".join(cell["source"])
        
        if "class AdvancedPortfolioEnv(gym.Env):" in text:
            # 1. Add peak_nav
            text = text.replace(
                "self.current_capital = self.initial_capital\n", 
                "self.current_capital = self.initial_capital\n            self.peak_nav = self.initial_capital\n"
            )
            
            # 2. Add Game Over Logic
            end_step_pattern = r'(\s*)(self\.current_step \+= 1\n\s*(?:done|terminated)\s*=\s*self\.current_step >= len\(self\.returns_arr\) - 1\n\s*return self\._get_obs\(\), reward, (?:done|terminated), False, \{\})'
            
            def repl(m):
                indent = m.group(1)
                
                game_over_logic = """
# ------------ HỆ THỐNG PHẠT DRAWDOWN & GAME OVER ------------
self.peak_nav = max(self.peak_nav, self.current_capital)
drawdown = (self.peak_nav - self.current_capital) / self.peak_nav

force_terminate = False
if drawdown > 0.05:
    reward -= 1000
if drawdown > 0.10:
    reward -= 5000
if drawdown > 0.15:
    reward -= 100000
    force_terminate = True
    if getattr(self, 'is_test', False):
        print(f"💀 GAME OVER: Cháy tài khoản! Drawdown {drawdown*100:.1f}% tại Step {self.current_step}")
# -----------------------------------------------------------

self.current_step += 1
terminated = (self.current_step >= len(self.returns_arr) - 1) or force_terminate
return self._get_obs(), float(reward), terminated, False, {}"""
                
                return "\n".join([indent + line if line.strip() else "" for line in game_over_logic.strip().split('\n')]) + "\n"
                
            new_text = re.sub(end_step_pattern, repl, text)
            if new_text == text: print("Failed to replace Game Over block!")
            text = new_text
            
            lines = [line + '\n' for line in text.split('\n')]
            if lines: lines[-1] = lines[-1][:-1]
            cell["source"] = lines

with open(r"c:\Users\ADMIN\Desktop\Kaggle\model\ppo_grouped_rl_v7.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Created ppo_grouped_rl_v7.ipynb with Game Over mechanics.")
