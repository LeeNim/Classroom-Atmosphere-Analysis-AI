import pandas as pd
import numpy as np
import time
import json
import os
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib

def main():
    # --- 1. Tải và tiền xử lý dữ liệu ---
    df = pd.read_csv('tonghop.csv')
    df.dropna(subset=['khong_khi_lop_hoc (label)'], inplace=True)

    features = [col for col in df.columns if col.startswith('percent_')]
    X = df[features]
    y = df['khong_khi_lop_hoc (label)']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
    )

    # --- 2. Tối ưu hóa và Huấn luyện mô hình ---
    print("--- Bắt đầu huấn luyện mô hình XGBoost (với GridSearchCV) ---")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.7, 0.8, 1.0]
    }

    grid_search = GridSearchCV(
        estimator=xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
        param_grid=param_grid,
        scoring='f1_macro',
        cv=3,
        verbose=1,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    print("✅ Huấn luyện và tối ưu hóa hoàn tất.")
    print("Tham số tốt nhất:", grid_search.best_params_)
    model = grid_search.best_estimator_

    # Lưu mô hình
    joblib.dump(model, 'model_xgboost.joblib')
    print("✅ Mô hình XGBoost đã được lưu vào 'model_xgboost.joblib'")

    # --- 3. Đánh giá mô hình ---
    start_time = time.time()
    y_pred = model.predict(X_test)
    end_time = time.time()
    prediction_time = end_time - start_time

    print("\n--- Báo cáo Phân loại (XGBoost) ---")
    report = classification_report(y_test, y_pred, target_names=le.classes_.astype(str), output_dict=True)
    print(classification_report(y_test, y_pred, target_names=le.classes_.astype(str)))
    
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - XGBoost")
    plt.savefig("cm_xgboost.png")
    print("Ma trận nhầm lẫn đã được lưu vào 'cm_xgboost.png'")

    # --- 4. Lưu kết quả để so sánh ---
    results = {
        'model': 'XGBoost',
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
    
    print(f"\n✅ Kết quả của XGBoost đã được lưu vào '{results_file}'")

if __name__ == '__main__':
    main()