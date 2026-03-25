import os
import numpy as np
import cv2

groups = ['A_pos', 'B_pos']
os.makedirs('dataset', exist_ok=True)

for group in groups:
    path = os.path.join('dataset', group)
    os.makedirs(path, exist_ok=True)
    for i in range(50):
        # Create a 128x128 random "fingerprint-like" noise image
        img = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
        cv2.imwrite(os.path.join(path, f'fake_{i}.jpg'), img)

print("Created 100 fake images in 'dataset/' folder!")
