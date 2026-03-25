import tensorflow as tf
import numpy as np
import sys

if len(sys.argv) < 2:
    print("Usage: python predict.py <image_path>")
    sys.exit()

model = tf.keras.models.load_model('blood_group_model.h5')
img = tf.keras.preprocessing.image.load_img(sys.argv[1], target_size=(128, 128), color_mode='grayscale')
img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
print(f"\nResulting Probabilities: {prediction}")
