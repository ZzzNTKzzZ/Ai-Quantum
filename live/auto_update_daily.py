import os
import subprocess
import datetime

script_dir = os.path.dirname(os.path.abspath(__file__))

def run_cmd(cmd, cwd=script_dir):
    print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] Đang chạy: {cmd}")
    res = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=False)
    if res.returncode != 0:
        print(f"LỖI KHI CHẠY: {cmd}")
        return False
    return True

def main():
    print("=====================================================")
    print("TỰ ĐỘNG CẬP NHẬT DỮ LIỆU CHỨNG KHOÁN - CHẾ ĐỘ LIVE")
    print("=====================================================")
    
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
