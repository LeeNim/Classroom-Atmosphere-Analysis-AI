# Classroom Atmosphere Analysis AI

An AI-powered system that analyzes classroom video footage to assess the overall emotional atmosphere and engagement level of students. This project uses computer vision to detect facial expressions and machine learning to classify the classroom environment as "Focused," "Lively," or "Distracted."

## Project Overview

In traditional classroom settings, it is challenging for educators to objectively and continuously gauge the collective emotional state of their students. This project addresses that challenge by providing a data-driven tool to analyze classroom dynamics. The system processes video recordings of a class, identifies student faces, recognizes their emotions, and aggregates this data to predict the overall "classroom atmosphere."

This tool can provide valuable feedback to educators, helping them understand student engagement and adjust their teaching methods accordingly.

## Key Features

- **Face Detection:** Utilizes the InsightFace model to accurately detect student faces, even in crowded classroom environments.
- **Emotion Recognition:** Employs the DeepFace library to classify facial expressions into seven core emotions (Happy, Sad, Angry, Fear, Surprise, Disgust, Neutral).
- **Mask Detection:** Includes a feature to identify whether students are wearing face masks. Emotion analysis is only performed on unmasked faces.
- **Two-Stage Architecture:**
    1.  **Feature Extraction Engine:** Processes a video file to analyze emotions frame-by-frame and outputs a `video_features.json` file containing the percentage distribution of all detected emotions.
    2.  **Classification Engine:** Takes the JSON file as input and uses a trained Multi-Layer Perceptron (MLP) neural network to predict the overall classroom atmosphere.
- **Comparative Analysis:** The MLP model was chosen after being benchmarked against Random Forest and XGBoost, demonstrating superior performance for this complex task.

## Technologies Used

- **Computer Vision:** OpenCV, InsightFace, DeepFace
- **Machine Learning:** TensorFlow, Keras, Scikit-learn, XGBoost
- **Data Handling:** NumPy, Pandas
- **Programming Language:** Python

## Project Architecture

The system is designed with a modular, two-stage architecture for efficiency and scalability.

### Stage 1: Feature Extraction Engine (`analysis.py`)

This engine handles the heavy computational tasks of video processing.

1.  **Input:** A single video file (`.mp4`).
2.  **Process:**
    -   The video is processed at a set interval (e.g., every 2-5 seconds).
    -   For each frame, InsightFace detects all faces and generates embedding vectors.
    -   A simple tracking algorithm assigns an ID to each face.
    -   A Keras model checks for face masks.
    -   For unmasked faces, DeepFace analyzes and classifies the dominant emotion.
3.  **Output:** A single `video_features.json` file that quantifies the emotional landscape of the entire video.

### Stage 2: Classification & Prediction Engine (`predict.py`)

This engine is lightweight and performs the final prediction.

1.  **Input:** The `video_features.json` file.
2.  **Process:**
    -   Loads the pre-trained MLP model (`model_mlp.keras`), scaler, and label encoder.
    -   The feature vector from the JSON file is scaled.
    -   The model predicts the classroom atmosphere and provides a confidence score.
3.  **Output:** A classification result, such as: `Classroom Atmosphere: Focused (Confidence: 85%)`.

## How to Use

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the analysis on a video:**
    ```bash
    python analysis.py --video path/to/your/video.mp4
    ```
4.  **Run the prediction:**
    ```bash
    python predict.py
    ```

## Future Development

-   **Improve Tracking Algorithm:** Integrate an advanced tracking model like DeepSORT to handle student movement and occlusions more robustly.
-   **Fine-Tune Emotion Model:** Use transfer learning to fine-tune the emotion recognition model on a custom dataset specific to educational contexts, helping it differentiate between states like "focused" and "sad."
-   **Multi-modal Analysis:** Incorporate other data streams, such as body posture analysis and head pose estimation, for a more holistic understanding of student engagement.
