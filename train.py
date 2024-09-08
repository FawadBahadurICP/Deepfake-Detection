import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.models import create_student_model, load_teacher_model, load_pretrained_teacher_model
from src.utils import save_plot, save_confusion_matrix
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load dataset
data = pd.read_csv('data/dataset.csv')
X = data.drop('class_label', axis=1).values
y = data['class_label'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Load and configure ResNet50 model as a teacher
teacher_model = load_pretrained_teacher_model('models/teacher.h5')  # Use the pre-trained ResNet50 model

# Create and train student model
student_model = create_student_model()
student_history = student_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)
student_model.save('models/student.h5')

# Plot and save results for student model
def save_plot(history, filename):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.title('Model Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label = 'val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.title('Model Loss')

    plt.savefig(filename)
    plt.close()

save_plot(student_history, 'results/student_accuracy.png')

# Evaluate student model
student_loss, student_accuracy = student_model.evaluate(X_test, y_test)

print(f'Student Model - Test Accuracy: {student_accuracy:.4f}, Test Loss: {student_loss:.4f}')

# Predict and save confusion matrix
y_pred_student = (student_model.predict(X_test) > 0.5).astype(int)
conf_matrix = confusion_matrix(y_test, y_pred_student)
disp = ConfusionMatrixDisplay(conf_matrix)
disp.plot()
plt.savefig('results/student_confusion_matrix.png')
