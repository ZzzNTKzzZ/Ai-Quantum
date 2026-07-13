import os
import sys
import datetime
import time
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_DIR = os.path.join(script_dir, "data", "stocks")
os.makedirs(RAW_DATA_DIR, exist_ok=True)

START_DATE = '2016-11-10'
END_DATE = '2026-06-23'
# if len(sys.argv) > 1:
#     END_DATE = sys.argv[1]
# else:
#     END_DATE = datetime.datetime.now().strftime('%Y-%m-%d')

MIN_SESSIONS = 1

def log(msg):
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")

def download_ticker(symbol):
    try:
        import vnstock
        out_path = os.path.join(RAW_DATA_DIR, f"{symbol}.csv")
        
        # 1. Lấy ngày cuối cùng từ file cũ (nếu có) để làm Start Date
        current_start_date = START_DATE
        df_existing = pd.DataFrame()
        
        if os.path.exists(out_path):
            df_existing = pd.read_csv(out_path)
            date_col_exist = 'time' if 'time' in df_existing.columns else ('Date' if 'Date' in df_existing.columns else None)
            if date_col_exist and not df_existing.empty:
                df_existing[date_col_exist] = pd.to_datetime(df_existing[date_col_exist])
                last_date = df_existing[date_col_exist].max()
                # Tiến thêm 1 ngày để lấy phần dữ liệu hoàn toàn mới
                current_start_date = (last_date + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
                
        # Kiểm tra nếu dữ liệu đã lấy kịch kim tới hiện tại rồi thì bỏ qua
        if pd.to_datetime(current_start_date) > pd.to_datetime(END_DATE):
            return {'symbol': symbol, 'status': 'UP-TO-DATE (Already latest)'}
            
        # Cố gắng lấy dữ liệu với các nguồn khác nhau
        sources = ['kbs', 'tcbs', 'vci']
        df_new = None
        
        for source in sources:
            try:
                time.sleep(3.5) # Tránh Rate Limit Exceeded
                q = vnstock.Quote(symbol=symbol, source=source)
                df_temp = q.history(start=current_start_date, end=END_DATE)
                if df_temp is not None and not df_temp.empty:
                    df_new = df_temp
                    break # Thành công thì thoát vòng lặp source
            except Exception as e:
                # Bỏ qua lỗi của source hiện tại, thử source tiếp theo
                pass
                
        if df_new is not None and not df_new.empty:
            date_col = 'time' if 'time' in df_new.columns else ('Date' if 'Date' in df_new.columns else None)
            if date_col:
                df_new[date_col] = pd.to_datetime(df_new[date_col])
                
                # Đổi tên cột về chuẩn 'time'
                if date_col != 'time':
                    df_new.rename(columns={date_col: 'time'}, inplace=True)
                    
                # 2. Nối dữ liệu mới vào dữ liệu cũ
                if not df_existing.empty:
                    if date_col_exist and date_col_exist != 'time':
                        df_existing.rename(columns={date_col_exist: 'time'}, inplace=True)
                    df_final = pd.concat([df_existing, df_new], ignore_index=True)
                else:
                    df_final = df_new
                    
                # Lọc trùng lặp, phòng hờ sàn trả về ngày trùng
                df_final = df_final.drop_duplicates(subset=['time'], keep='last')
                df_final = df_final.sort_values('time')
                
                # Lưu file CSV riêng cho từng mã vào thư mục live/data/stocks/
                df_final.to_csv(out_path, index=False)
                
                return {'symbol': symbol, 'status': f'SUCCESS (Added {len(df_new)} rows)'}
                
        # Xử lý trường hợp có file nhưng API không trả về dữ liệu mới (do nghỉ lễ, cuối tuần, hoặc kbs trả về rỗng)
        if not df_existing.empty:
            return {'symbol': symbol, 'status': 'UP-TO-DATE (No new trading sessions)'}
            
        return {'symbol': symbol, 'status': 'EMPTY'}
    except Exception as e:
        # Bắt lỗi toàn cục
        return {'symbol': symbol, 'status': 'ERROR', 'error': str(e)}

def main():
    log("==========================================")
    log("BẮT ĐẦU CRAWL 46 MÃ CHO CHẾ ĐỘ LIVE")
    log("==========================================")
    
    # Chỉ lấy đúng 46 mã
    all_symbols = ['BID', 'BMP', 'BVH', 'CII', 'CTD', 'CTG', 'DCM', 'DGW', 'DIG', 'DPM', 'DXG', 'EIB', 'FPT', 'GAS', 'GMD', 'HAG', 'HCM', 'HDC', 'HPG', 'HSG', 'HT1', 'KBC', 'KDC', 'KDH', 'MBB', 'MSN', 'MWG', 'NKG', 'NLG', 'NT2', 'PDR', 'PHR', 'PNJ', 'PVD', 'PVT', 'REE', 'SBT', 'SJS', 'SSI', 'STB', 'TCH', 'VCB', 'VHC', 'VIC', 'VNM', 'VSC']
    log(f"Đã nạp {len(all_symbols)} mã chứng khoán.")
    
    for symbol in all_symbols:
        log(f"Đang tải dữ liệu cho {symbol}...")
        res = download_ticker(symbol)
        if res['status'] == 'SUCCESS':
            log(f" -> Thành công!")
        else:
            log(f" -> Lỗi/Trống: {res}")
            
    log("==========================================")
    log(f"HOÀN TẤT TẢI DỮ LIỆU. ĐÃ LƯU TẠI: {RAW_DATA_DIR}")
    log("==========================================")

if __name__ == "__main__":
    main()
