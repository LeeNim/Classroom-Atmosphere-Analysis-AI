import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace
from insightface.app import FaceAnalysis
import tensorflow as tf
from tf_keras.models import load_model
from collections import Counter
import json
import os
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. Cấu hình ---

# THAY ĐỔI Ở ĐÂY: Liệt kê tên các thư mục chứa video clip
CAC_THU_MUC_VIDEO = ["MAH06917_opencv", "MAH07303_opencv", "video_opencv"]

# Tên file CSV đầu ra
OUTPUT_CSV_PATH = "tong_hop_phan_tich.csv"

# Đường dẫn đến mô hình phát hiện khẩu trang
MASK_MODEL_PATH = 'model-facemask.h5'

# Tham số cấu hình
ANALYSIS_INTERVAL_SEC = 2
CONFIDENCE_THRESHOLD = 0.5
MASK_PROB_THRESHOLD = 0.7
FACE_TRACKING_THRESHOLD = 0.3

# --- 2. Tải các mô hình ---
print("Đang tải các mô hình...")

# Tải mô hình InsightFace với OpenVINO cho GPU Intel
try:
    app = FaceAnalysis(name='buffalo_l', 
                       providers=['OpenVINOExecutionProvider', 'CPUExecutionProvider']) 
    app.prepare(ctx_id=0, det_size=(640, 640)) 
    print("✅ InsightFace đã tải và cấu hình cho GPU Intel (OpenVINO).")
except Exception as e:
    print(f"Lỗi khi tải InsightFace với OpenVINO: {e}.")
    print("Thử lại với CPU...")
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("✅ InsightFace đã tải trên CPU.")

# Tải mô hình phát hiện khẩu trang Keras
try:
    mask_model = load_model(MASK_MODEL_PATH)
    mask_model_input_size = (mask_model.input_shape[1], mask_model.input_shape[2])
    print(f"✅ Mô hình phát hiện khẩu trang đã tải.")
except Exception as e:
    print(f"Lỗi khi tải mô hình khẩu trang từ {MASK_MODEL_PATH}: {e}.")
    exit()

print("--- Hoàn tất tải mô hình ---")


# --- 3. Hàm phân tích video chuyên dụng ---

def predict_mask(face_img, input_size):
    """Dự đoán xác suất đeo khẩu trang từ ảnh khuôn mặt."""
    face_img = cv2.resize(face_img, input_size)
    face_img = np.expand_dims(face_img, axis=0)
    face_img = face_img / 255.0 
    prediction = mask_model.predict(face_img, verbose=0)[0][0]
    return prediction

def phan_tich_video(video_path):
    """
    Phân tích một file video clip và trả về một dictionary chứa kết quả tổng hợp.
    """
    print(f"\n▶️ Bắt đầu phân tích video: '{video_path}'")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Lỗi: Không thể mở file video {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print(f"  Lỗi: FPS của video {video_path} bằng 0.")
        cap.release()
        return None

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Các biến này được khởi tạo lại cho mỗi video để đảm bảo theo dõi độc lập
    all_results_this_clip = []
    tracked_students = {}
    next_student_id = 1
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Chỉ phân tích các frame theo khoảng thời gian đã định
        if frame_count % int(fps * ANALYSIS_INTERVAL_SEC) == 0:
            current_time_sec = frame_count / fps
            
            # =================================================================
            # === BẮT ĐẦU LOGIC PHÂN TÍCH ĐẦY ĐỦ TỪ CODE GỐC CỦA BẠN ===
            # =================================================================
            faces = app.get(frame) 
            current_frame_faces_data = [] 

            for i, face in enumerate(faces):
                bbox = face.bbox.astype(np.int32)
                x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
                x, y = max(0, x), max(0, y)
                w, h = min(w, frame_width - x), min(h, frame_height - y)

                if w <= 0 or h <= 0: continue

                face_img_cropped = frame[y:y+h, x:x+w]
                if face_img_cropped.size == 0: continue
                
                deepface_input_size = (224, 224)
                face_resized_for_deepface = cv2.resize(face_img_cropped, deepface_input_size, interpolation=cv2.INTER_AREA)

                mask_prob = predict_mask(face_resized_for_deepface, mask_model_input_size)
                is_masked = mask_prob > MASK_PROB_THRESHOLD
                
                emotion, confidence = "N/A", 0.0
                
                if not is_masked:
                    try:
                        analysis = DeepFace.analyze(face_resized_for_deepface, actions=['emotion'], enforce_detection=False, silent=True)
                        if isinstance(analysis, list) and analysis:
                            dominant_emotion = analysis[0]['dominant_emotion']
                            emotion_scores = analysis[0]['emotion']
                            confidence = emotion_scores.get(dominant_emotion, 0) / 100.0
                            
                            if confidence >= CONFIDENCE_THRESHOLD:
                                emotion = dominant_emotion
                            else:
                                emotion = "unknown"
                    except Exception as e:
                        emotion = "error_analysis"
                else:
                    emotion = "masked"

                face_embedding = face.embedding if hasattr(face, 'embedding') else (face.normed_embedding if hasattr(face, 'normed_embedding') else None)

                current_face_data = {
                    'bbox': bbox.tolist(),
                    'emotion': emotion,
                    'confidence': float(confidence),
                    'mask_prob': float(mask_prob),
                    'student_id': 'unknown', 
                    'embedding': face_embedding.tolist() if face_embedding is not None else None
                }
                current_frame_faces_data.append(current_face_data)

            # --- Theo dõi học sinh (gán ID) ---
            unassigned_faces = [f for f in current_frame_faces_data if f['embedding'] is not None]
            assigned_faces_in_frame = [f for f in current_frame_faces_data if f['embedding'] is None]

            for student_id, student_info in list(tracked_students.items()):
                if 'last_embedding' in student_info and student_info['last_embedding'] is not None:
                    similarities = [cosine_similarity(np.array(student_info['last_embedding']).reshape(1, -1), np.array(face_data['embedding']).reshape(1, -1))[0][0] for face_data in unassigned_faces]
                    if similarities:
                        max_similarity_idx = np.argmax(similarities)
                        if similarities[max_similarity_idx] > FACE_TRACKING_THRESHOLD:
                            assigned_face = unassigned_faces.pop(max_similarity_idx)
                            assigned_face['student_id'] = student_id
                            assigned_faces_in_frame.append(assigned_face)
                            student_info['last_embedding'] = assigned_face['embedding']
                            student_info['emotions'].append({'time': current_time_sec, 'emotion': assigned_face['emotion']})
            
            for new_face_data in unassigned_faces:
                new_face_data['student_id'] = next_student_id
                assigned_faces_in_frame.append(new_face_data)
                tracked_students[next_student_id] = {
                    'last_embedding': new_face_data['embedding'],
                    'emotions': [{'time': current_time_sec, 'emotion': new_face_data['emotion']}]
                }
                next_student_id += 1

            all_results_this_clip.extend(assigned_faces_in_frame)
            # ===============================================================
            # === KẾT THÚC LOGIC PHÂN TÍCH ĐẦY ĐỦ TỪ CODE GỐC CỦA BẠN ===
            # ===============================================================

        frame_count += 1
    
    cap.release()
    
    # --- Tổng hợp kết quả cho video clip này ---
    if not all_results_this_clip:
        print(f"  Cảnh báo: Không có khuôn mặt nào được phát hiện trong video '{video_path}'.")
        return None

    invalid_emotion_labels = ['masked', 'unknown', 'N/A', 'error_analysis', 'temp_untracked']
    overall_valid_emotions = [res['emotion'] for res in all_results_this_clip if res['emotion'] not in invalid_emotion_labels]
    
    if not overall_valid_emotions:
        return {
            'video_path': video_path,
            'total_valid_emotion_detections': 0,
            'dominant_emotion': 'N/A',
        }

    overall_emotion_counts = Counter(overall_valid_emotions)
    total_overall_valid_emotions = len(overall_valid_emotions)
    
    overall_emotion_percentages = {}
    # Tạo các cột phần trăm cho tất cả các cảm xúc có thể có
    possible_emotions = ['happy', 'sad', 'angry', 'surprise', 'fear', 'disgust', 'neutral']
    for emotion in possible_emotions:
        count = overall_emotion_counts.get(emotion, 0)
        overall_emotion_percentages[f"percent_{emotion}"] = round((count / total_overall_valid_emotions) * 100, 2)

    summary = {
        'video_path': video_path,
        'total_valid_emotion_detections': total_overall_valid_emotions,
        'dominant_emotion': max(overall_emotion_counts, key=overall_emotion_counts.get),
        **overall_emotion_percentages  # Gộp dict percentages vào summary
    }
    
    print(f"  ✅ Hoàn tất phân tích: '{video_path}'")
    return summary


# --- 4. Chương trình chính ---
if __name__ == "__main__":
    tong_hop_ket_qua = []

    # Vòng lặp qua các thư mục được chỉ định
    for ten_thu_muc in CAC_THU_MUC_VIDEO:
        if not os.path.isdir(ten_thu_muc):
            print(f"Cảnh báo: Bỏ qua vì không tìm thấy thư mục '{ten_thu_muc}'")
            continue

        print(f"\n📁 Đang xử lý thư mục: '{ten_thu_muc}'")
        
        cac_file_video = sorted([f for f in os.listdir(ten_thu_muc) if f.lower().endswith('.mp4')])

        for ten_file in cac_file_video:
            duong_dan_video_day_du = os.path.join(ten_thu_muc, ten_file)
            ket_qua_clip = phan_tich_video(duong_dan_video_day_du)
            
            if ket_qua_clip:
                tong_hop_ket_qua.append(ket_qua_clip)
                
                # ===================================================
                # === THAY ĐỔI MỚI: In DataFrame tạm thời ra màn hình ===
                # ===================================================
                print("\n--- Bảng kết quả tạm thời ---")
                df_tam_thoi = pd.DataFrame(tong_hop_ket_qua)
                df_tam_thoi.fillna(0, inplace=True) # Điền 0 vào các ô trống cho đẹp
                print(df_tam_thoi.to_string()) # Dùng to_string() để hiển thị tất cả các cột
                print("---------------------------------")
                # ===================================================

    # --- 5. Xuất kết quả tổng hợp ra file CSV ---
    if not tong_hop_ket_qua:
        print("\nKhông có kết quả nào để xuất ra file CSV.")
    else:
        print(f"\n📊 Đang tổng hợp kết quả từ {len(tong_hop_ket_qua)} video clips...")
        
        df = pd.DataFrame(tong_hop_ket_qua)
        df.fillna(0, inplace=True)
        
        cols_order = ['video_path', 'total_valid_emotion_detections', 'dominant_emotion']
        percent_cols = sorted([col for col in df.columns if col.startswith('percent_')])
        final_cols = cols_order + percent_cols
        # Đảm bảo chỉ lấy các cột có trong DataFrame để tránh lỗi
        final_cols = [col for col in final_cols if col in df.columns]
        df = df[final_cols]
        
        df['khong_khi_lop_hoc (label)'] = ''
        
        df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')
        
        print(f"🎉 Hoàn tất! Đã lưu kết quả tổng hợp vào file '{OUTPUT_CSV_PATH}'.")
        print("Bây giờ bạn có thể mở file CSV và điền vào cột cuối cùng để gán nhãn.")