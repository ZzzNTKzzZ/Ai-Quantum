import os
import shutil

# Đường dẫn tương đối an toàn dựa trên vị trí file script
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))
src_dir = os.path.join(root_dir, "data", "raw")
dst_dir = os.path.join(root_dir, "data", "processed")

mappings = {
    "c1_amihud_illiq.csv": "amihud_illiq.csv",
    "c2_return_dispersion.csv": "return_dispersion.csv",
    "e1_usdvnd.csv": "usdvnd.csv",
    "e2_vnibor_overnight.csv": "vnibor_overnight.csv",
    "e3_us10y_yield.csv": "tpcp_5y_yield.csv",
    "e4_epu.csv": "epu.csv",
    "e5_gpr.csv": "gpr.csv",
    "e6_m2_vietnam.csv": "m2_vietnam.csv",
    "e7_credit_growth_vn.csv": "credit_growth_vn.csv",
    "e8_cpi_vietnam.csv": "cpi_vietnam.csv",
    "g1_dxy.csv": "dxy.csv",
    "g2_fed_funds_rate.csv": "fed_funds_rate.csv",
    "g3_us10y_yield.csv": "us10y_yield.csv",
    "g4_china_sse.csv": "china_sse.csv",
    "g5_fdi_vietnam.csv": "fdi_vietnam.csv",
    "m2_vix.csv": "vix.csv",
    "m3_sp500.csv": "sp500.csv",
    "m4_foreign_net_buy_sell.csv": "foreign_net_buy_sell.csv",
    "m5_hose_volume.csv": "hose_volume.csv",
    "s1_brent_oil.csv": "brent_oil.csv",
    "s2_gold.csv": "gold.csv",
    "s3_pmi_vietnam.csv": "pmi_vietnam.csv",
    "s4_copper_price.csv": "copper_price.csv"
}

for dst_file, src_file in mappings.items():
    src_path = os.path.join(src_dir, src_file)
    dst_path = os.path.join(dst_dir, dst_file)
    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
        print(f"Copied {src_path} -> {dst_path}")
    else:
        print(f"Warning: {src_path} does not exist.")
