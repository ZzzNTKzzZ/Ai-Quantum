import os
import sys
import glob

sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import time
import requests

def get_future_prices_vndirect(ticker, start_date, days=3):
    """
    Lấy giá đóng cửa từ API VNDirect cho N ngày giao dịch tiếp theo.
    """
    end_date = (datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=10)).strftime("%Y-%m-%d")
    url = f"https://finfo-api.vndirect.com.vn/v4/stock_prices?sort=date&q=code:{ticker}~date:gte:{start_date}~date:lte:{end_date}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        data = response.json()
        if data and 'data' in data and len(data['data']) >= days:
            prices = []
            for i in range(days):
                close_price = data['data'][i].get('adClose') or data['data'][i].get('close')
                prices.append(close_price)
            return prices
    except Exception as e:
        pass
    return None

def get_future_prices_yfinance(ticker, start_date, days=3):
    """
    Lấy giá đóng cửa từ Yahoo Finance cho N ngày giao dịch tiếp theo.
    """
    yf_ticker = f"{ticker}.VN"
    end_date = (datetime.strptime(start_date, "%Y-%m-%d") + timedelta(days=10)).strftime("%Y-%m-%d")
    try:
        stock = yf.Ticker(yf_ticker)
        hist = stock.history(start=start_date, end=end_date)
        if len(hist) >= days:
            prices = hist['Close'].iloc[:days].tolist()
            return [p / 1000 for p in prices]
    except Exception as e:
        pass
    return None

def evaluate_predictions():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output")
    
    # Tìm file CSV mới nhất
    signal_files = glob.glob(os.path.join(output_dir, "live_trading_signals_*.csv"))
    if not signal_files:
        print("❌ Không tìm thấy file live_trading_signals_*.csv nào trong thư mục output.")
        return
        
    latest_csv = max(signal_files, key=os.path.getctime)
    print(f"Đang phân tích file tín hiệu mới nhất: {os.path.basename(latest_csv)}")
    
    df = pd.read_csv(latest_csv)
    
    # Lấy ngày từ tên file
    date_str_raw = os.path.basename(latest_csv).split('_')[-1].split('.')[0]
    date_to_check = datetime.strptime(date_str_raw, "%Y%m%d").strftime("%Y-%m-%d")
    
    print(f"Bắt đầu đánh giá dự báo từ ngày: {date_to_check}\n")

    results_list = []
    
    for index, row in df.iterrows():
        ticker = row['ticker']
        signal = row['Tín Hiệu']
        
        is_buy_signal = "Mua" in signal or "Tăng" in signal
        is_sell_signal = "Bán" in signal or "Giảm" in signal
        
        if not is_buy_signal and not is_sell_signal:
            continue
            
        prices = get_future_prices_vndirect(ticker, date_to_check, days=3)
        if prices is None:
            prices = get_future_prices_yfinance(ticker, date_to_check, days=3)
            
        if prices is None or len(prices) < 3:
            continue
            
        t0_price, t1_price, t2_price = prices[0], prices[1], prices[2]
        
        if t0_price > 1000:
            t0_price, t1_price, t2_price = t0_price/1000, t1_price/1000, t2_price/1000
            
        pct_change_t2 = ((t2_price - t0_price) / t0_price) * 100
        
        if is_buy_signal:
            is_correct = pct_change_t2 > 0
        elif is_sell_signal:
            is_correct = pct_change_t2 < 0
        else:
            is_correct = False
            
        signal_short = "MUA" if is_buy_signal else "BÁN/CẢNH BÁO"
        result_text = "ĐÚNG" if is_correct else "SAI"
        
        results_list.append({
            "Mã": ticker,
            "Tín Hiệu": signal_short,
            "Giá T0": round(t0_price, 2),
            "Giá T+1": round(t1_price, 2),
            "Giá T+2": round(t2_price, 2),
            "Lãi/Lỗ % (T+2)": round(pct_change_t2, 2),
            "Đánh Giá": result_text
        })
        
        time.sleep(0.2)
        
    if not results_list:
        print("Không có mã nào đủ dữ liệu T+2 để đánh giá.")
        return
        
    leaderboard_df = pd.DataFrame(results_list)
    # Sắp xếp Leaderboard: Lãi nhiều nhất xếp trên cùng
    leaderboard_df = leaderboard_df.sort_values(by="Lãi/Lỗ % (T+2)", ascending=False).reset_index(drop=True)
    
    print("\n" + "="*50)
    print("🏆 LEADERBOARD HIỆU QUẢ DỰ BÁO (T+2)")
    print("="*50)
    print(leaderboard_df.to_string(index=False))
    
    total = len(leaderboard_df)
    correct = len(leaderboard_df[leaderboard_df["Đánh Giá"] == "ĐÚNG"])
    win_rate = (correct / total) * 100
    
    print("\n" + "="*50)
    print(f"Tổng số mã được đánh giá : {total}")
    print(f"Số dự báo ĐÚNG            : {correct}")
    print(f"Tỷ lệ chiến thắng (Win rate): {win_rate:.2f}%")
    print("="*50)
    
    leaderboard_csv = os.path.join(output_dir, f"leaderboard_verification_{date_str_raw}.csv")
    leaderboard_df.to_csv(leaderboard_csv, index=False, encoding='utf-8-sig')
    print(f"\n[OK] Đã xuất Leaderboard ra file: {leaderboard_csv}")

if __name__ == "__main__":
    evaluate_predictions()
