import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

def plot_comparison(results_file='comparison_results.json'):
    """Đọc file kết quả và vẽ biểu đồ so sánh các mô hình."""
    try:
        df = pd.read_json(results_file)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file '{results_file}'. Vui lòng chạy các script huấn luyện trước.")
        return

    print("--- Dữ liệu so sánh ---")
    print(df)

    # --- Biểu đồ 1: So sánh Độ chính xác và F1-Score ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Dùng melt để chuyển đổi dataframe cho dễ vẽ
    df_melted = df.melt(
        id_vars='model', 
        value_vars=['accuracy', 'f1_score_macro'],
        var_name='metric', 
        value_name='score'
    )

    sns.barplot(data=df_melted, x='model', y='score', hue='metric', ax=ax1, palette='viridis')
    ax1.set_title('So sánh Hiệu suất các Mô hình', fontsize=16, fontweight='bold')
    ax1.set_xlabel('Mô hình', fontsize=12)
    ax1.set_ylabel('Điểm số', fontsize=12)
    ax1.set_ylim(0, 1.1)
    
    # Thêm giá trị trên các cột
    for p in ax1.patches:
        ax1.annotate(f'{p.get_height():.3f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', xytext=(0, 9), textcoords='offset points')

    plt.legend(title='Chỉ số')
    plt.tight_layout()
    plt.savefig('model_performance_comparison.png')
    print("\nBiểu đồ so sánh hiệu suất đã lưu vào 'model_performance_comparison.png'")
    plt.show()

    # --- Biểu đồ 2: So sánh Thời gian dự đoán ---
    fig, ax2 = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df, x='model', y='prediction_time_ms', ax=ax2, palette='plasma')
    ax2.set_title('So sánh Thời gian Dự đoán (trên tập Test)', fontsize=16, fontweight='bold')
    ax2.set_xlabel('Mô hình', fontsize=12)
    ax2.set_ylabel('Thời gian (ms)', fontsize=12)
    
    for p in ax2.patches:
        ax2.annotate(f'{p.get_height():.2f} ms', (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='center', xytext=(0, 9), textcoords='offset points')

    plt.tight_layout()
    plt.savefig('model_time_comparison.png')
    print("Biểu đồ so sánh thời gian đã lưu vào 'model_time_comparison.png'")
    plt.show()


if __name__ == '__main__':
    plot_comparison()