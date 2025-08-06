import cv2
import numpy as np
from deepface import DeepFace
from insightface.app import FaceAnalysis
import tensorflow as tf
from collections import Counter
import os
import json

# --- 1. CẤU HÌNH ---
VIDEO_PATH = "MAH07303.mp4" # <--- ĐƯỜNG DẪN VIDEO ĐẦU VÀO
OUTPUT_JSON_PATH = "video_features.json"
MASK_MODEL_PATH = 'model-facemask.h5'

# Tham số phân tích
ANALYSIS_INTERVAL_SEC = 2
MASK_PROB_THRESHOLD = 0.7

# --- 2. TẢI CÁC MÔ HÌNH PHÂN TÍCH ---
print("Đang tải các mô hình phân tích (InsightFace, DeepFace, Mask)...")
try:
    # Vẫn tải các model này bằng tf.keras để xem lỗi có xảy ra ở đây không
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    mask_model = tf.keras.models.load_model(MASK_MODEL_PATH)
    print("✅ Các mô hình phân tích đã tải thành công.")
except Exception as e:
    print(f"Lỗi khi tải mô hình phân tích: {e}")
    exit()

# --- 3. PHÂN TÍCH VIDEO VÀ TRÍCH XUẤT ĐẶC TRƯNG ---
print(f"\n▶️ Bắt đầu phân tích video '{VIDEO_PATH}' để trích xuất đặc trưng...")
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    print(f"Lỗi: Không đọc được FPS từ video {VIDEO_PATH}")
    exit()

all_emotions_in_clip = []
frame_count = 0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    if frame_count % int(fps * ANALYSIS_INTERVAL_SEC) == 0:
        try:
            faces = app.get(frame)
            for face in faces:
                bbox = face.bbox.astype(np.int32)
                face_img_cropped = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                if face_img_cropped.size == 0: continue
                
                face_resized = cv2.resize(face_img_cropped, (224, 224), interpolation=cv2.INTER_AREA)
                face_batch = np.expand_dims(face_resized / 255.0, axis=0)
                mask_prob = mask_model.predict(face_batch, verbose=0)[0][0]
                
                emotion = "masked"
                if mask_prob < MASK_PROB_THRESHOLD:
                    results = DeepFace.analyze(face_resized, actions=['emotion'], enforce_detection=False, silent=True)
                    emotion = results[0]['dominant_emotion']
                
                all_emotions_in_clip.append(emotion)
        except Exception:
            pass
    
    frame_count += 1
    print(f"\rĐã xử lý frame: {frame_count}/{total_frames}", end="")

cap.release()
print("\n✅ Phân tích và thu thập dữ liệu hoàn tất.")

# --- 4. TÍNH TOÁN VÀ LƯU ĐẶC TRƯNG RA FILE JSON ---
print(f"\n💾 Đang tính toán và lưu đặc trưng vào file '{OUTPUT_JSON_PATH}'...")
output_data = {'video_path': VIDEO_PATH, 'features': {}}
if all_emotions_in_clip:
    valid_emotions = [e for e in all_emotions_in_clip if e not in ['masked', 'unknown', 'error']]
    emotion_counts = Counter(valid_emotions)
    total_valid = sum(emotion_counts.values())
    
    if total_valid > 0:
        possible_emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        for emotion in sorted(possible_emotions):
            percent = (emotion_counts.get(emotion, 0) / total_valid * 100)
            output_data['features'][f"percent_{emotion}"] = percent

with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=4)

print("🎉 Hoàn tất! Đã tạo file đặc trưng. Bây giờ hãy chạy file 'predict_from_json.py'.")