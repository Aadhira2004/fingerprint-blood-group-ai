import tensorflow as tf
import numpy as np
import cv2
import os
import sys

# 1. Load the best model
model = tf.keras.models.load_model('blood_group_model.keras')

# 2. Function to predict
def predict_and_show(img_path):
    # Load and preprocess
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (128, 128))
    img_normalized = img_resized / 255.0
    img_final = np.expand_dims(np.expand_dims(img_normalized, axis=0), axis=-1)

    # Predict
    prediction = model.predict(img_final)
    class_idx = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    # Get labels (assuming A_pos and B_pos)
    labels = ['A_Positive', 'B_Positive'] # Adjust based on your folder names
    result = labels[class_idx]

    print(f"\n[IMAGE]: {img_path}")
    print(f"[PREDICTION]: {result}")
    print(f"[CONFIDENCE]: {confidence:.2f}%")

# Test on one image from your validation set
# Change 'dataset/A_pos/fake_0.jpg' to an actual filename you have
predict_and_show('dataset/A_pos/fake_0.jpg')
