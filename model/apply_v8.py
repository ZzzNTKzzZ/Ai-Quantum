import json

with open(r"c:\Users\ADMIN\Desktop\Kaggle\model\ppo_grouped_rl_v5.ipynb", "r", encoding="utf-8") as f:
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
            end_step_old = """            self.current_step += 1
            done = self.current_step >= self.n_steps - 1 
            
            return self._get_obs(), float(reward), done, False, {'net_return': daily_ret}"""
            
            end_step_new = """            # ------------ HỆ THỐNG PHẠT DRAWDOWN & GAME OVER ------------
            self.peak_nav = max(self.peak_nav, getattr(self, 'current_capital', 100_000_000))
            drawdown = (self.peak_nav - getattr(self, 'current_capital', 100_000_000)) / self.peak_nav
            
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
            done = (self.current_step >= self.n_steps - 1) or force_terminate
            
            return self._get_obs(), float(reward), done, False, {'net_return': daily_ret}"""
            
            text = text.replace(end_step_old, end_step_new)
            
            lines = [line + '\n' for line in text.split('\n')]
            if lines: lines[-1] = lines[-1][:-1]
            cell["source"] = lines

with open(r"c:\Users\ADMIN\Desktop\Kaggle\model\ppo_grouped_rl_v8.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Created ppo_grouped_rl_v8.ipynb successfully.")
