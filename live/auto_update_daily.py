import os
import subprocess
import datetime
import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))

def run_cmd(cmd, cwd=script_dir):
    print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] Đang chạy: {cmd}")
    res = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=False)
    if res.returncode != 0:
        print(f"LỖI KHI CHẠY: {cmd}")
        return False
    return True

def check_and_update_macro():
    raw_dir = os.path.join(script_dir, "data", "raw")
    test_dir = os.path.join(script_dir, "data", "test")
    macro_file = os.path.join(raw_dir, "usdvnd.csv")
    
    needs_update = True
    today_str = '2026-06-26' #  datetime.datetime.now().strftime('%Y-%m-%d')
    start_date_str = "2016-11-10"
    
    if os.path.exists(macro_file):
        try:
            df = pd.read_csv(macro_file)
            date_col = 'Date' if 'Date' in df.columns else 'time'
            df[date_col] = pd.to_datetime(df[date_col])
            max_date = df[date_col].max()
            
            # Nếu chênh lệch <= 3 ngày (tính cả T7, CN) thì không cần crawl lại
            if (datetime.datetime.now() - max_date).days <= 3:
                needs_update = False
                print(f"\n[MACRO] Dữ liệu vĩ mô khá mới (Cập nhật đến: {max_date.strftime('%Y-%m-%d')}). Bỏ qua bước crawl vĩ mô.")
            else:
                # Chỉ lấy dữ liệu từ ngày cuối cùng có trong data cũ
                start_date_str = max_date.strftime('%Y-%m-%d')
        except Exception as e:
            print(f"\n[MACRO] Lỗi đọc {macro_file}: {e}")
            
    if needs_update:
        print(f"\n[MACRO] Tự động crawl dữ liệu mới từ {start_date_str} đến {today_str}...")
        cmd = f"python src/data_collection/crawl_macro_data.py {start_date_str} {today_str}"
        if run_cmd(cmd):
            print("\n[MACRO] Đang tiến hành ghép (Merge) dữ liệu mới vào lịch sử cũ...")
            for f in os.listdir(test_dir):
                if not f.endswith('.csv'): continue
                
                test_path = os.path.join(test_dir, f)
                raw_path = os.path.join(raw_dir, f)
                
                if os.path.exists(raw_path):
                    try:
                        df_new = pd.read_csv(test_path)
                        df_old = pd.read_csv(raw_path)
                        df_combined = pd.concat([df_old, df_new], ignore_index=True)
                        
                        # Tìm cột Date để xóa trùng
                        date_cands = [c for c in df_combined.columns if c.lower() in ['date', 'time', 'observation_date', 'quarter']]
                        if date_cands:
                            dc = date_cands[0]
                            df_combined = df_combined.drop_duplicates(subset=[dc], keep='last')
                            # Sort theo chuỗi vẫn chính xác do định dạng chuẩn ISO (YYYY-MM-DD)
                            df_combined = df_combined.sort_values(dc)
                            
                        df_combined.to_csv(test_path, index=False)
                    except Exception as e:
                        print(f"[MACRO] Lỗi khi merge file {f}: {e}")
            print("[MACRO] Merge thành công! Dữ liệu đã liền mạch.")

def main():
    print("=====================================================")
    print("TỰ ĐỘNG CẬP NHẬT DỮ LIỆU CHỨNG KHOÁN - CHẾ ĐỘ LIVE")
    print("=====================================================")
    
    # 0. Kiểm tra và tải Dữ liệu Vĩ mô tự động
    check_and_update_macro()
    
    # 1. Tải dữ liệu 46 mã mới nhất bằng crawler đặc biệt cho Live
    import sys
    crawl_cmd = f"python crawl_live_46.py {sys.argv[1]}" if len(sys.argv) > 1 else "python crawl_live_46.py"
    if not run_cmd(crawl_cmd): return
    
    # 2. Chạy toàn bộ pipeline xử lý dữ liệu (tạo m1_vn46.csv, căn chỉnh, v.v.)
    if not run_cmd("python src/data_processing/run_full_regeneration.py"): return
    
    # 3. Cập nhật HMM bằng chế độ LIVE (Tự động Load Model, không cần train lại)
    print("\n[HMM] Đang chạy cập nhật xác suất Trạng thái thị trường (Fast Mode)...")
    if not run_cmd("python hmm_live_inference.py"): return

    # 4. Chạy mô hình dự báo Mua/Bán (Live Trading T+1)
    print("\n[LightGBM] Đang huấn luyện và xuất tín hiệu giao dịch...")
    if not run_cmd("python live_trading.py"): return

    # 5. DRL Live Trading (Sinh tỷ trọng)
    print("\n[DRL] Đang chạy PPO Meta-Agent để trích xuất tỷ trọng giao dịch...")
    if not run_cmd("python drl_live_trading.py"): return

    print("\n=====================================================")
    print("HOÀN TẤT TẢI VÀ TIỀN XỬ LÝ DỮ LIỆU!")
    print("Tín hiệu dự báo giao dịch ngày mai đã được lưu thành công trong folder 'live/output'.")
    print("=====================================================")

if __name__ == "__main__":
    main()
