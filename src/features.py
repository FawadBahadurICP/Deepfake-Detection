import numpy as np
import dlib
import cv2

# Initialize face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')

def compute_features(landmarks_points):
    # Extract landmarks
    left_eye = landmarks_points[36:42]
    right_eye = landmarks_points[42:48]
    mouth = landmarks_points[48:68]
    nose = landmarks_points[27:36]
    eyebrows = landmarks_points[17:27]
    
    # Compute feature values
    face_width = np.linalg.norm(landmarks_points[16] - landmarks_points[0])
    face_height = np.linalg.norm(landmarks_points[8] - landmarks_points[19])
    nose_length = np.linalg.norm(landmarks_points[27] - landmarks_points[33])
    eyebrow_distance = np.mean([np.linalg.norm(eyebrows[i] - eyebrows[i+1]) for i in range(0, len(eyebrows)-1, 2)])
    
    # Eye Aspect Ratio (EAR) calculation
    def compute_ear(eye_points):
        A = np.linalg.norm(eye_points[1] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])
        C = np.linalg.norm(eye_points[0] - eye_points[3])
        return (A + B) / (2.0 * C)
    
    ear_left = compute_ear(left_eye)
    ear_right = compute_ear(right_eye)
    
    # Facial symmetry
    left_face = np.mean(landmarks_points[0:17], axis=0)
    right_face = np.mean(landmarks_points[17:26], axis=0)
    facial_symmetry = np.linalg.norm(left_face - right_face)
    
    # Contour measurements
    contour_length = np.sum([np.linalg.norm(landmarks_points[i] - landmarks_points[i+1]) for i in range(len(landmarks_points) - 1)])
    
    return [face_width, face_height, nose_length, eyebrow_distance, ear_left, ear_right, facial_symmetry, contour_length]

def extract_geometric_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    features = []
    for face in faces:
        landmarks = predictor(gray, face)
        landmarks_points = np.array([(p.x, p.y) for p in landmarks.parts()])
        features.append(compute_features(landmarks_points))
    
    return np.array(features).mean(axis=0) if features else np.zeros(8)
