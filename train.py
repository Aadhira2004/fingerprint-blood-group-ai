import tensorflow as tf
from tensorflow.keras import layers, models, Input, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

IMG_SIZE = (128, 128)
BATCH_SIZE = 8 # Smaller batch size for better stability

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,
    zoom_range=0.05
)

train_gen = datagen.flow_from_directory(
    'dataset', target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', subset='training', color_mode='grayscale'
)

val_gen = datagen.flow_from_directory(
    'dataset', target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', subset='validation', color_mode='grayscale'
)

model = models.Sequential([
    Input(shape=(128, 128, 1)),
    # Using HeNormal initialization for more stable start
    layers.Conv2D(16, (3, 3), activation='leaky_relu', kernel_initializer='he_normal'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(32, (3, 3), activation='leaky_relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(64, activation='leaky_relu'),
    layers.Dropout(0.4),
    layers.Dense(len(train_gen.class_indices), activation='softmax')
])

# Very slow learning rate to prevent exploding loss
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

# Stop training if the validation loss doesn't improve for 5 epochs
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

print("\n🚀 Training with Stability Patch...")
model.fit(train_gen, validation_data=val_gen, epochs=30, callbacks=[early_stop])

model.save('blood_group_model.keras')
print("\n[+] Success! Optimized model saved.")
