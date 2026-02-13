import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import glob

# SKLearn for Classical ML and Metrics
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

# TensorFlow for Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==========================================
# 1. DATASET HANDLING & GENERATION
# ==========================================
def create_dummy_data(base_path):
    """Creates dummy images to allow this code to run for demonstration."""
    classes = ['Glass', 'Paper', 'Cardboard', 'Plastic', 'Metal', 'Trash']
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    for cls in classes:
        cls_path = os.path.join(base_path, cls)
        os.makedirs(cls_path, exist_ok=True)
        # Create 40 random images per class (Total ~240 images > 200 requirement)
        for i in range(40): 
            # Random noise image
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(cls_path, f'{cls}_{i}.jpg'), img)
    
    print(f"Dummy dataset created at {base_path}")
    return classes

DATASET_PATH = "./dataset_trashnet"
CLASSES = create_dummy_data(DATASET_PATH)

# ==========================================
# 2. DATA INGESTION & PREPROCESSING
# ==========================================
print("\n--- 2. Loading and Preprocessing Data ---")

data = []
labels = []
IMAGE_SIZE = (224, 224)

for category in CLASSES:
    path = os.path.join(DATASET_PATH, category)
    class_num = CLASSES.index(category)
    for img_name in os.listdir(path):
        try:
            img_path = os.path.join(path, img_name)
            img_arr = cv2.imread(img_path)
            # Resize
            img_resized = cv2.resize(img_arr, IMAGE_SIZE)
            data.append(img_resized)
            labels.append(class_num)
        except Exception as e:
            continue

data = np.array(data)
labels = np.array(labels)

# Normalize Data (Scale 0-1)
data_normalized = data / 255.0

print(f"Data Shape: {data.shape}")
print(f"Labels Shape: {labels.shape}")

# Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(data_normalized, labels, test_size=0.2, random_state=42)
print(f"Training Samples: {X_train.shape[0]}")
print(f"Testing Samples: {X_test.shape[0]}")

# ==========================================
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ==========================================
print("\n--- 3. Exploratory Data Analysis ---")

# A. Numeric Analysis: Class Distribution
unique, counts = np.unique(labels, return_counts=True)
dist_dict = dict(zip(CLASSES, counts))
print("Class Distribution:", dist_dict)

# B. Visual Analysis: Bar Chart
plt.figure(figsize=(8, 5))
sns.barplot(x=list(dist_dict.keys()), y=list(dist_dict.values()), palette='viridis')
plt.title("Class Distribution in Dataset")
plt.xlabel("Waste Category")
plt.ylabel("Count")
plt.show()

# C. Visual Analysis: Sample Images
plt.figure(figsize=(10, 5))
for i in range(6):
    plt.subplot(1, 6, i+1)
    # Grab first image of each class (logic simplified for demo)
    idx = np.where(labels == i)[0][0]
    plt.imshow(cv2.cvtColor(data[idx], cv2.COLOR_BGR2RGB))
    plt.title(CLASSES[i])
    plt.axis('off')
plt.suptitle("Sample Images from Dataset")
plt.show()

# ==========================================
# 4. MODEL SELECTION & TRAINING
# ==========================================

# --- APPROACH A: SVM (From PDF List) ---
print("\n--- Model A: Support Vector Machine (SVM) ---")

# Flatten data for SVM (Samples, Height*Width*Channels)
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

# Pipeline: PCA (Dimensionality Reduction) -> SVM
svm_pipeline = Pipeline([
    ('pca', PCA(n_components=50, whiten=True)), # Reduce to 50 features
    ('svc', SVC(kernel='rbf', class_weight='balanced'))
])

svm_pipeline.fit(X_train_flat, y_train)
svm_preds = svm_pipeline.predict(X_test_flat)

# --- APPROACH B: CNN (Model of Own Choice) ---
print("\n--- Model B: Convolutional Neural Network (CNN) ---")

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(CLASSES), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train CNN
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=1)

# ==========================================
# 5. EVALUATION
# ==========================================
print("\n--- 5. Final Evaluation & Comparison ---")

def evaluate_model(y_true, y_pred, model_name):
    print(f"\nResults for {model_name}:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASSES, zero_division=0))
    
    # Confusion Matrix Plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

# Evaluate SVM
evaluate_model(y_test, svm_preds, "SVM (with PCA)")

# Evaluate CNN
cnn_probs = model.predict(X_test)
cnn_preds = np.argmax(cnn_probs, axis=1)
evaluate_model(y_test, cnn_preds, "CNN (Deep Learning)")

# Comparative Plot
svm_acc = accuracy_score(y_test, svm_preds)
cnn_acc = accuracy_score(y_test, cnn_preds)

plt.figure(figsize=(5, 4))
plt.bar(['SVM', 'CNN'], [svm_acc, cnn_acc], color=['skyblue', 'salmon'])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy Score")
plt.show()

print("\nPipeline Execution Complete.")