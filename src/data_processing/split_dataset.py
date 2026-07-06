import pandas as pd
import os
import sys

# Thiết lập encoding UTF-8 cho console để tránh lỗi hiển thị tiếng Việt trên Windows
sys.stdout.reconfigure(encoding='utf-8')

def split_and_save_dataset():
    # Đường dẫn thư mục dự án
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))
    
    # Sử dụng đúng đường dẫn đầu ra mới 'output/hmm' của pipeline đã tối ưu
    input_path = os.path.join(root_dir, "output", "hmm", "master_drl_ready.parquet")
    output_dir = os.path.join(root_dir, "output", "hmm", "splits")
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(input_path):
        print(f"Lỗi: Không tìm thấy tệp nguồn {input_path}")
        return
        
    print(f"Đang đọc dữ liệu từ {input_path}...")
    df = pd.read_parquet(input_path)
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').reset_index(drop=True)
    
    # Định nghĩa mốc thời gian phân chia (Time-Series Split)
    train_end = '2019-12-31'
    val_end = '2022-12-31'
    
    df_train = df[df['time'] <= train_end].reset_index(drop=True)
    df_val = df[(df['time'] > train_end) & (df['time'] <= val_end)].reset_index(drop=True)
    df_test = df[df['time'] > val_end].reset_index(drop=True)
    
    print("\n--- Kết quả phân chia tập dữ liệu ---")
    print(f"1. Tập Train      : Từ {df_train['time'].min().strftime('%Y-%m-%d')} đến {df_train['time'].max().strftime('%Y-%m-%d')} | Kích thước: {df_train.shape}")
    print(f"2. Tập Validation : Từ {df_val['time'].min().strftime('%Y-%m-%d')} đến {df_val['time'].max().strftime('%Y-%m-%d')} | Kích thước: {df_val.shape}")
    print(f"3. Tập Test       : Từ {df_test['time'].min().strftime('%Y-%m-%d')} đến {df_test['time'].max().strftime('%Y-%m-%d')} | Kích thước: {df_test.shape}")
    
    splits = {
        "train_set": df_train,
        "val_set": df_val,
        "test_set": df_test
    }
    
    print(f"\nĐang lưu các tập dữ liệu vào thư mục riêng: {output_dir}")
    for name, split_df in splits.items():
        parquet_out = os.path.join(output_dir, f"{name}.parquet")
        csv_out = os.path.join(output_dir, f"{name}.csv")
        
        # Lưu định dạng Parquet
        split_df.to_parquet(parquet_out, index=False)
        # Lưu định dạng CSV
        split_df.to_csv(csv_out, index=False)
        
        print(f" - Đã lưu {name}.parquet & {name}.csv")
        
    print("\nHội tụ thành công! Thư mục splits đã chứa đầy đủ 6 tệp dữ liệu.")

if __name__ == "__main__":
    split_and_save_dataset()
