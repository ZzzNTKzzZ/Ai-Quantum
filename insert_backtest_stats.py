import nbformat as nbf
import os

d_path = 'c:/Users/ADMIN/Desktop/Kaggle/notebooks/d.ipynb'

with open(d_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

# Find where section 10 starts to insert before it
idx_10 = -1
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'markdown' and '10. Lưu Kết Quả' in cell.source:
        idx_10 = i
        break

if idx_10 == -1:
    idx_10 = len(nb.cells)

# Create 9.3 Markdown
md_9_3 = """### 9.3 Thống kê Hiệu suất Tài chính (Financial Backtest)
**Mục đích:** Mô phỏng giao dịch thực tế trên tập Walk-Forward OOS. So sánh Lợi nhuận tích lũy (Equity Curve) và Sharpe Ratio của Chiến lược so với việc Mua & Nắm giữ toàn bộ rổ cổ phiếu.
**Quy tắc giả lập:** 
- Vốn được chia đều cho tất cả các mã (Ví dụ rổ có 46 mã, mỗi mã chiếm 1/46 vốn).
- Ngày T: Nếu Meta-Classifier dự báo `prob > 0.5`, ta mua/nắm giữ mã đó ở ngày T+1. Nếu `prob <= 0.5`, bán ra giữ tiền mặt (lợi suất = 0)."""

# Create 9.3 Code
code_9_3 = """
import numpy as np
import matplotlib.pyplot as plt

if 'df_backtest' in locals() and 'final_meta_pred_prob' in df_backtest.columns:
    df_bt = df_backtest.dropna(subset=['final_meta_pred_prob']).copy()
    
    # Tạo tín hiệu mua (1) và đứng ngoài (0)
    df_bt['signal'] = (df_bt['final_meta_pred_prob'] > 0.5).astype(int)
    
    # Tính lợi nhuận danh mục (Equal Weighting)
    # Giả định vốn chia đều cho N mã. Lệnh mua ăn target_return_1d, lệnh đứng ngoài ăn 0%
    N_stocks = df_bt['ticker'].nunique()
    
    portfolio = df_bt.groupby('time').apply(
        lambda x: (x['target_return_1d'] * x['signal']).sum() / N_stocks
    ).reset_index(name='port_ret')
    
    # Benchmark: Mua và nắm giữ toàn bộ rổ cổ phiếu
    benchmark = df_bt.groupby('time').apply(
        lambda x: x['target_return_1d'].sum() / N_stocks
    ).reset_index(name='bench_ret')
    
    df_perf = pd.merge(portfolio, benchmark, on='time')
    df_perf['port_cum'] = (1 + df_perf['port_ret']).cumprod()
    df_perf['bench_cum'] = (1 + df_perf['bench_ret']).cumprod()
    
    # Hàm tính Metrics
    def calc_metrics(returns):
        cum = (1 + returns).cumprod()
        ann_ret = (cum.iloc[-1] ** (252 / len(returns))) - 1
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        roll_max = cum.cummax()
        drawdown = cum / roll_max - 1
        max_dd = drawdown.min()
        win_rate = (returns > 0).mean()
        return ann_ret, ann_vol, sharpe, max_dd, win_rate

    p_ret, p_vol, p_shp, p_dd, p_win = calc_metrics(df_perf['port_ret'])
    b_ret, b_vol, b_shp, b_dd, b_win = calc_metrics(df_perf['bench_ret'])
    
    # Hiển thị Bảng Thống Kê
    perf_table = pd.DataFrame({
        'Chiến lược HMM+LGBM': [f"{p_ret*100:.1f}%", f"{p_vol*100:.1f}%", f"{p_shp:.2f}", f"{p_dd*100:.1f}%", f"{p_win*100:.1f}%"],
        'Thị trường (Buy&Hold)': [f"{b_ret*100:.1f}%", f"{b_vol*100:.1f}%", f"{b_shp:.2f}", f"{b_dd*100:.1f}%", f"{b_win*100:.1f}%"]
    }, index=['Lợi nhuận năm (Ann. Ret)', 'Biến động năm (Ann. Vol)', 'Sharpe Ratio', 'Max Drawdown', 'Tỷ lệ ngày lãi (Win Rate)'])
    
    print("\\n📊 BẢNG THỐNG KÊ HIỆU SUẤT GIAO DỊCH (OUT-OF-SAMPLE)")
    display(perf_table)
    
    # Vẽ Equity Curve
    plt.figure(figsize=(14, 6))
    plt.plot(df_perf['time'], df_perf['port_cum'], label=f"Chiến lược HMM+LGBM (Sharpe: {p_shp:.2f})", color='#e74c3c', linewidth=2)
    plt.plot(df_perf['time'], df_perf['bench_cum'], label=f"Thị trường (Sharpe: {b_shp:.2f})", color='#95a5a6', linewidth=1.5, alpha=0.8)
    
    plt.title("📈 Đường cong Lợi nhuận (Equity Curve) - Walk-Forward Backtest", fontsize=16, fontweight='bold')
    plt.ylabel("Tài khoản (Tỉ lệ)", fontsize=12)
    plt.xlabel("Thời gian", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.fill_between(df_perf['time'], df_perf['port_cum'], df_perf['bench_cum'], where=(df_perf['port_cum'] > df_perf['bench_cum']), color='#2ecc71', alpha=0.1)
    plt.tight_layout()
    plt.show()
else:
    print("Vui lòng chạy Ô 9.1 (Walk-Forward Backtest) trước để có dữ liệu vẽ biểu đồ!")
"""

# Insert the cells
cells_to_insert = [
    nbf.v4.new_markdown_cell(md_9_3),
    nbf.v4.new_code_cell(code_9_3)
]

nb.cells = nb.cells[:idx_10] + cells_to_insert + nb.cells[idx_10:]

with open(d_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Inserted Section 9.3 Backtest Metrics into d.ipynb")
