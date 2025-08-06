import subprocess
import sys
import os

# --- CẤU HÌNH ---
# Tên của hai file script cần chạy
script_phan_tich = "analysis.py"
script_du_doan = "predict.py"

def run_script(script_name):
    """Hàm để chạy một script Python và kiểm tra lỗi."""
    # Kiểm tra xem file script có tồn tại không
    if not os.path.exists(script_name):
        print(f"❌ Lỗi: Không tìm thấy file script '{script_name}'.")
        return False
        
    print("\n" + "="*50)
    print(f"▶️  Bắt đầu chạy script: {script_name}")
    print("="*50)
    
    try:
        # sys.executable đảm bảo dùng đúng trình thông dịch Python đang chạy file này
        # check=True sẽ báo lỗi nếu script con thất bại
        subprocess.run([sys.executable, script_name], check=True)
        print(f"\n✅ Script '{script_name}' đã chạy thành công.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Lỗi khi chạy script '{script_name}'. Dừng quy trình.")
        return False
    except Exception as e:
        print(f"\n❌ Đã xảy ra lỗi không xác định: {e}")
        return False

if __name__ == "__main__":
    # Chạy script phân tích trước
    success = run_script(script_phan_tich)
    
    # Nếu script phân tích thành công, tiếp tục chạy script dự đoán
    if success:
        run_script(script_du_doan)
    
    print("\n🎉 Quy trình đã hoàn tất.")