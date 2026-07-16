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
            # Lấy N ngày giao dịch đầu tiên
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
            # Chia 1000 để chuẩn hóa giá VNĐ
            return [p / 1000 for p in prices]
    except Exception as e:
        pass
    return None

def evaluate_predictions():
    csv_file = r"c:\Users\ADMIN\Desktop\Kaggle\live\output\live_trading_signals_20260618.csv"
    
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"❌ Không tìm thấy file: {csv_file}")
        return

    date_to_check = "2026-06-20"
    print(f"Bắt đầu đánh giá dự báo cho tín hiệu ngày: {date_to_check}\n")
    print(f"{'Mã':<5} | {'Tín Hiệu':<18} | {'Giá T0':<7} | {'Giá T+1':<7} | {'Giá T+2':<7} | {'Lãi/Lỗ (T+2)':<12} | {'Kết Quả'}")
    print("-" * 80)

    correct_predictions = 0
    total_evaluated = 0
    
    for index, row in df.iterrows():
        ticker = row['ticker']
        signal = row['Tín Hiệu']
        
        # Chỉ đánh giá các mã có khuyến nghị Mua hoặc Bán rõ ràng
        is_buy_signal = "Mua" in signal or "Tăng" in signal
        is_sell_signal = "Bán" in signal or "Giảm" in signal
        
        if not is_buy_signal and not is_sell_signal:
            continue
            
        prices = get_future_prices_vndirect(ticker, date_to_check, days=3)
        if prices is None:
            prices = get_future_prices_yfinance(ticker, date_to_check, days=3)
            
        if prices is None or len(prices) < 3:
            # print(f"⚠️ [{ticker}] Không đủ dữ liệu T+2 để đánh giá.")
            continue
            
        t0_price, t1_price, t2_price = prices[0], prices[1], prices[2]
        
        if t0_price > 1000:
            t0_price, t1_price, t2_price = t0_price/1000, t1_price/1000, t2_price/1000
            
        # Tính % thay đổi so với T0
        pct_change_t2 = ((t2_price - t0_price) / t0_price) * 100
        
        # Đánh giá đúng sai
        if is_buy_signal:
            is_correct = pct_change_t2 > 0
        elif is_sell_signal:
            is_correct = pct_change_t2 < 0
        else:
            is_correct = False
            
        result_text = "✅ ĐÚNG" if is_correct else "❌ SAI"
        if is_correct:
            correct_predictions += 1
            
        total_evaluated += 1
        
        # Định dạng output
        signal_short = "MUA" if is_buy_signal else "BÁN/CẢNH BÁO"
        color_change = f"+{pct_change_t2:.2f}%" if pct_change_t2 > 0 else f"{pct_change_t2:.2f}%"
        
        print(f"{ticker:<5} | {signal_short:<18} | {t0_price:<7.2f} | {t1_price:<7.2f} | {t2_price:<7.2f} | {color_change:<12} | {result_text}")
        
        time.sleep(0.2) # Tránh bị block API

    print("\n" + "="*40)
    print("TỔNG KẾT HIỆU QUẢ DỰ BÁO (T+2)")
    print("="*40)
    print(f"Tổng số mã được đánh giá : {total_evaluated}")
    if total_evaluated > 0:
        win_rate = (correct_predictions / total_evaluated) * 100
        print(f"Số dự báo ĐÚNG            : {correct_predictions}")
        print(f"Tỷ lệ chiến thắng (Win rate): {win_rate:.2f}%")
    else:
        print("Không có mã nào được đánh giá.")

if __name__ == "__main__":
    evaluate_predictions()
