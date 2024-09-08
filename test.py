import cv2
import numpy as np
from tensorflow.keras.models import load_model
from src.features import extract_geometric_features
from src.video_processing import extract_frames
from sklearn.preprocessing import StandardScaler

# Load model and scaler
model = load_model('models/student.h5')  # Change to 'teacher.h5' if testing teacher model

# Create a scaler for the feature data
scaler = StandardScaler()

def predict_video(video_path):
    frames = extract_frames(video_path)
    features = [extract_geometric_features(frame) for frame in frames]
    
    if len(features) == 0:
        return 'No face detected'
    
    features = np.mean(features, axis=0).reshape(1, -1)
    features = scaler.fit_transform(features)  # Ensure features are scaled

    prediction = model.predict(features)
    return 'Fake' if prediction[0] < 0.5 else 'Real'

# Test with a sample video
video_path = 'path/to/test_video.mp4'
print(predict_video(video_path))
