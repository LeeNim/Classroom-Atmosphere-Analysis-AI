import json
import numpy as np
import tensorflow as tf
import joblib
import os

# --- 1. CẤU HÌNH ---
INPUT_JSON_PATH = "video_features.json"
MLP_MODEL_PATH = 'model_mlp.keras'
SCALER_PATH = 'scaler.joblib'
LABEL_ENCODER_PATH = 'label_encoder.joblib'

# --- 2. TẢI MÔ HÌNH PHÂN LOẠI VÀ CÁC CÔNG CỤ ---
print("Đang tải mô hình MLP, Scaler và LabelEncoder...")
# Kiểm tra xem các file cần thiết có tồn tại không
required_files = [INPUT_JSON_PATH, MLP_MODEL_PATH, SCALER_PATH, LABEL_ENCODER_PATH]
for file_path in required_files:
    if not os.path.exists(file_path):
        print(f"Lỗi: Không tìm thấy file '{file_path}'. Vui lòng chạy script training và analysis trước.")
        exit()

try:
    # SỬ DỤNG tf.keras.models.load_model CHÍNH THỨC
    final_model = tf.keras.models.load_model(MLP_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    print("✅ Mô hình và công cụ đã tải thành công.")
except Exception as e:
    print(f"Lỗi khi tải mô hình: {e}")
    print("Đây chính là lỗi 'batch_shape' bạn gặp. Vấn đề 100% nằm ở file model hoặc môi trường TensorFlow.")
    exit()

# --- 3. ĐỌC DỮ LIỆU TỪ JSON VÀ DỰ ĐOÁN ---
print(f"\n🧠 Đang đọc đặc trưng từ '{INPUT_JSON_PATH}' và dự đoán...")
with open(INPUT_JSON_PATH, 'r') as f:
    data = json.load(f)

feature_dict = data.get('features', {})
if not feature_dict:
    print("Lỗi: File JSON không chứa đặc trưng để dự đoán.")
    exit()

# Sắp xếp các feature theo thứ tự alphabet để đảm bảo nhất quán
features = [feature_dict.get(f"percent_{emotion}", 0) for emotion in sorted(['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise'])]

features_scaled = scaler.transform(np.array(features).reshape(1, -1))

prediction_proba = final_model.predict(features_scaled)[0]
prediction_index = np.argmax(prediction_proba)
prediction_label = label_encoder.inverse_transform([prediction_index])[0]

print("\n" + "="*40)
print(f" KẾT QUẢ DỰ ĐOÁN CHO VIDEO: {data['video_path']}")
print("="*40)
print(f"  ▶️ Không khí lớp học: {str(prediction_label)}")
print(f"  ▶️ Độ tin cậy (Confidence): {prediction_proba[prediction_index]:.2%}")
print("="*40)