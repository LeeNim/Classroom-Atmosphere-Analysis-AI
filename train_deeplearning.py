from operator import le
import pandas as pd
import numpy as np
import time
import json
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib

def build_mlp_model(input_shape, num_classes):
    """Xây dựng mô hình MLP với Dropout."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_shape,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main():
    # --- 1. Tải và tiền xử lý dữ liệu ---
    df = pd.read_csv('tonghop.csv')
    df.dropna(subset=['khong_khi_lop_hoc (label)'], inplace=True)

    features = [col for col in df.columns if col.startswith('percent_')]
    X = df[features]
    y = df['khong_khi_lop_hoc (label)']

    # Chuyển đổi nhãn text/số về dạng 0, 1, 2...
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)

    # Phân chia dữ liệu
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
    )

    # Chuẩn hóa dữ liệu - Rất quan trọng cho Deep Learning
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- 2. Xây dựng và Huấn luyện mô hình ---
    print("--- Bắt đầu huấn luyện mô hình Deep Learning (MLP) ---")
    model = build_mlp_model(X_train_scaled.shape[1], num_classes)
    model.summary()
    history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=8, validation_split=0.2, verbose=2)
    print("✅ Huấn luyện hoàn tất.")

    # Lưu mô hình
    model.save('model_mlp.keras')
    print("✅ Mô hình Deep Learning đã được lưu vào 'model_mlp2.keras'")
    joblib.dump(scaler, 'scaler.joblib')
    joblib.dump(le, 'label_encoder.joblib')
    print("✅ Scaler và LabelEncoder đã được lưu vào 'scaler.joblib' và 'label_encoder.joblib'")

    # --- 3. Đánh giá mô hình ---
    start_time = time.time()
    y_pred_proba = model.predict(X_test_scaled)
    y_pred = np.argmax(y_pred_proba, axis=1)
    end_time = time.time()
    prediction_time = end_time - start_time

    print("\n--- Báo cáo Phân loại (Deep Learning) ---")
    report = classification_report(y_test, y_pred, target_names=le.classes_.astype(str), output_dict=True)
    print(classification_report(y_test, y_pred, target_names=le.classes_.astype(str)))
    
    # Vẽ và lưu ma trận nhầm lẫn
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - Deep Learning (MLP)")
    plt.savefig("cm_deep_learning.png")
    print("Ma trận nhầm lẫn đã được lưu vào 'cm_deep_learning.png'")

    # --- 4. Lưu kết quả để so sánh ---
    results = {
        'model': 'Deep Learning (MLP)',
        'accuracy': report['accuracy'],
        'f1_score_macro': report['macro avg']['f1-score'],
        'prediction_time_ms': prediction_time * 1000
    }

    results_file = 'comparison_results.json'
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = []
    
    all_results.append(results)
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print(f"\n✅ Kết quả của Deep Learning đã được lưu vào '{results_file}'")

if __name__ == '__main__':
    main()