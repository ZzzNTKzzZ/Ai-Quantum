with open('README.md', 'a', encoding='utf-8') as f:
    f.write('\n\n## 🚀 Kiến trúc Mới Nhất: Hierarchical Dual-Frequency HMM (`f.ipynb`)\n')
    f.write('Hệ thống HMM đã được tái cấu trúc hoàn toàn trong `notebooks/f.ipynb` nhằm khắc phục lỗi Look-ahead Bias và nhiễu hàm bậc thang (Step-function noise) khi gộp dữ liệu Tháng và Ngày.\n')
    f.write('- **Tầng 1 (Macro HMM - Khung Tháng):** Xử lý độc lập dữ liệu vĩ mô theo từng tháng, xuất ra `Macro_Prob` và tịnh tiến (shift) lùi 1 tháng để áp dụng cho trading thực tế.\n')
    f.write('- **Tầng 2 (Market HMM - Khung Ngày):** Đánh giá thị trường hàng ngày dựa trên các chỉ báo kỹ thuật ngày kết hợp với `Macro_Prob` từ Tầng 1.\n')
    f.write('- **Tầng 3 (Ticker HMM):** Tích hợp cả Xác suất Vĩ mô, Xác suất Thị trường, Xác suất Dòng tiền Ngành và đặc trưng riêng của Ticker để dán nhãn trạng thái tối thượng.\n')
