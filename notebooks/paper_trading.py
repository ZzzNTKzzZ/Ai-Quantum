#!/usr/bin/env python
# coding: utf-8

# # 📈 Hệ Thống Paper Trading (Giao Dịch Ảo)
# 
# Notebook này được thiết kế để nạp lại bộ não AI (mô hình RL PPO) mới nhất từ file `.zip` và mô phỏng giao dịch thực chiến (Paper Trading). Bạn có thể kiểm thử danh mục đầu tư với vốn ảo trước khi áp dụng vào tài khoản thật.

# In[9]:


import os
import sys
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

# Thêm thư mục gốc vào sys.path để có thể import từ src và model
sys.path.append(os.path.abspath('..'))

from model.ppo_grouped_rl import CONFIG, AdvancedTickerExtractor, load_data, AdvancedPortfolioEnv

# print("✅ Khởi tạo môi trường thành công!")


# ## 1. Nạp Mô Hình AI Mới Nhất & Dữ Liệu

# In[ ]:


# Nạp dữ liệu thị trường mới nhất
print("⏳ Đang lấy dữ liệu thị trường mới nhất...")
returns_df, ai_features_df, strategies_features_df, weights_dim, tickers, num_strategies_features, dates = load_data()

# Khởi tạo môi trường ảo để format input cho AI
def make_env():
    return AdvancedPortfolioEnv(
        returns_df=returns_df,
        ai_features_df=ai_features_df,
        strategies_features_df=strategies_features_df,
        weights_dim=weights_dim,
        tickers=tickers,
        num_strategies_features=num_strategies_features,
        window_size=getattr(CONFIG, 'TEST_WINDOW', 60)
    )

env = DummyVecEnv([make_env])

# Nạp VecNormalize (Cực kỳ quan trọng để chuẩn hóa input)
NORMALIZE_PATH = '../model/v8/vec_normalize.pkl' # Đổi theo version
if os.path.exists(NORMALIZE_PATH):
    env = VecNormalize.load(NORMALIZE_PATH, env)
    env.training = False
    env.norm_reward = False
    print("✅ Đã nạp thành công bộ chuẩn hóa (VecNormalize)!")

# Nạp mô hình AI
MODEL_PATH = '../model/v8/AI_Brain_v8.zip' # Đổi theo version
print(f"🔄 Đang nạp bộ não AI từ: {MODEL_PATH} ...")
model = PPO.load(MODEL_PATH)
print("✅ Đã nạp thành công bộ não AI!")


# ## 2. Dự Báo Danh Mục & Paper Trading
# Chạy mô hình AI trên dữ liệu hôm nay để lấy danh mục (Portfolio Weights) cho ngày mai.

# In[ ]:


VON_KHOI_DIEM = 100_000_000  # 100 triệu VND vốn ảo

def paper_trading_predict(env, tickers, capital):
    """Dự báo hành động và in ra lệnh giao dịch cụ thể"""
    obs = env.reset()
    print("🧠 AI đang tính toán tỷ trọng danh mục tối ưu...")

    # AI Dự báo tỷ trọng
    action, _states = model.predict(obs, deterministic=True)
    action = action[0] # Lấy action từ DummyVecEnv

    print(f"\n{'='*40}")
    print(f"🎯 ĐỀ XUẤT GIAO DỊCH CHO NGÀY T+1")
    print(f"{'='*40}")
    print(f"Vốn hiện tại: {capital:,.0f} VND\n")

    for idx, ticker in enumerate(tickers):
        weight = action[idx]
        if weight > 0.01: # Bỏ qua các mã có tỷ trọng < 1%
            allocated_cash = capital * weight
            print(f"> 🟢 MUA/NẮM GIỮ {ticker}: {weight*100:5.2f}% danh mục (tương đương {allocated_cash:,.0f} VND)")

    print(f"\n📌 Ghi chú: Sử dụng các tỷ lệ này để đặt lệnh trên hệ thống Paper Trading (ví dụ: FireAnt, SSI iBoard ảo).")

# Chạy dự báo
paper_trading_predict(env=env, tickers=tickers, capital=VON_KHOI_DIEM)

