import nbformat as nbf

d_path = 'c:/Users/ADMIN/Desktop/Kaggle/notebooks/d.ipynb'

with open(d_path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

new_code_9_3 = """
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

if 'df_backtest' in locals() and 'final_meta_pred_prob' in df_backtest.columns:
    df_bt = df_backtest.dropna(subset=['final_meta_pred_prob']).copy()
    N_stocks = df_bt['ticker'].nunique()
    
    # 1. Chiến lược 1: Mua rải đều (AI Equal Weight)
    # Lệnh mua ăn target_return_1d, lệnh đứng ngoài ăn 0%
    df_bt['signal_eq'] = (df_bt['final_meta_pred_prob'] > 0.5).astype(int)
    port_eq = df_bt.groupby('time').apply(
        lambda x: (x['target_return_1d'] * x['signal_eq']).sum() / N_stocks
    ).reset_index(name='ret_eq')
    
    # 2. Chiến lược 2: Tập trung Top 5 (AI Top 5 Conviction)
    # Mỗi ngày chọn ra 5 mã có xác suất cao nhất. Nếu xác suất > 0.5 thì mua (dành 20% vốn).
    def top_k_return(day_data, k=5):
        # Sắp xếp từ cao xuống thấp
        top_k = day_data.sort_values('final_meta_pred_prob', ascending=False).head(k)
        # Chỉ mua những mã > 0.5
        top_k_buy = top_k[top_k['final_meta_pred_prob'] > 0.5]
        if len(top_k_buy) == 0:
            return 0.0
        # Tính trung bình lợi nhuận của k mã (vốn chia đều k phần)
        return top_k_buy['target_return_1d'].sum() / k

    port_top5 = df_bt.groupby('time').apply(lambda x: top_k_return(x, k=5)).reset_index(name='ret_top5')
    
    # 3. Chiến lược Benchmark (Buy & Hold toàn bộ rổ)
    bench = df_bt.groupby('time').apply(
        lambda x: x['target_return_1d'].sum() / N_stocks
    ).reset_index(name='ret_bench')
    
    # Gộp kết quả
    df_perf = pd.merge(port_eq, port_top5, on='time')
    df_perf = pd.merge(df_perf, bench, on='time')
    
    df_perf['cum_eq'] = (1 + df_perf['ret_eq']).cumprod()
    df_perf['cum_top5'] = (1 + df_perf['ret_top5']).cumprod()
    df_perf['cum_bench'] = (1 + df_perf['ret_bench']).cumprod()
    
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

    metrics_eq = calc_metrics(df_perf['ret_eq'])
    metrics_top5 = calc_metrics(df_perf['ret_top5'])
    metrics_bench = calc_metrics(df_perf['ret_bench'])
    
    # Hiển thị Bảng Thống Kê
    perf_table = pd.DataFrame({
        'AI Tập Trung (Top 5)': [f"{metrics_top5[0]*100:.1f}%", f"{metrics_top5[1]*100:.1f}%", f"{metrics_top5[2]:.2f}", f"{metrics_top5[3]*100:.1f}%", f"{metrics_top5[4]*100:.1f}%"],
        'AI Rải Đều (Equal)': [f"{metrics_eq[0]*100:.1f}%", f"{metrics_eq[1]*100:.1f}%", f"{metrics_eq[2]:.2f}", f"{metrics_eq[3]*100:.1f}%", f"{metrics_eq[4]*100:.1f}%"],
        'Buy & Hold Benchmark': [f"{metrics_bench[0]*100:.1f}%", f"{metrics_bench[1]*100:.1f}%", f"{metrics_bench[2]:.2f}", f"{metrics_bench[3]*100:.1f}%", f"{metrics_bench[4]*100:.1f}%"]
    }, index=['Lợi nhuận năm (Ann. Ret)', 'Biến động năm (Ann. Vol)', 'Sharpe Ratio', 'Max Drawdown', 'Tỷ lệ ngày lãi (Win Rate)'])
    
    print("\\n📊 BẢNG THỐNG KÊ HIỆU SUẤT GIAO DỊCH VỚI NHIỀU CHIẾN THUẬT (OOS)")
    display(perf_table)
    
    # Vẽ Equity Curve
    plt.figure(figsize=(14, 7))
    plt.plot(df_perf['time'], df_perf['cum_top5'], label=f"AI Tập Trung Top 5 (Sharpe: {metrics_top5[2]:.2f})", color='#f39c12', linewidth=2.5)
    plt.plot(df_perf['time'], df_perf['cum_eq'], label=f"AI Rải Đều (Sharpe: {metrics_eq[2]:.2f})", color='#e74c3c', linewidth=2)
    plt.plot(df_perf['time'], df_perf['cum_bench'], label=f"Thị trường (Sharpe: {metrics_bench[2]:.2f})", color='#95a5a6', linewidth=1.5, alpha=0.8)
    
    plt.title("📈 Đường cong Lợi nhuận (Equity Curve) - Đa Chiến Thuật", fontsize=16, fontweight='bold')
    plt.ylabel("Tài khoản (Tỉ lệ)", fontsize=12)
    plt.xlabel("Thời gian", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
else:
    print("Vui lòng chạy Ô 9.1 (Walk-Forward Backtest) trước để có dữ liệu vẽ biểu đồ!")
"""

found = False
for cell in nb.cells:
    if cell.cell_type == 'code' and "df_bt = df_backtest.dropna" in cell.source and "port_cum" in cell.source:
        cell.source = new_code_9_3
        found = True
        break

if found:
    with open(d_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print("Successfully added Top-5 strategy.")
else:
    print("Could not find the target cell.")
