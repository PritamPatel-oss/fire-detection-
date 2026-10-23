import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# =============================================
# CONFIGURATION
# =============================================
DATASET_PATH = '../dataset'  # Create: dataset/fire/ and dataset/no_fire/
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 15
MODEL_SAVE_PATH = "fire_model.h5"

# =============================================
# DATA GENERATORS
# =============================================
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    rotation_range=15
)

train_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_data = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# =============================================
# CNN MODEL
# =============================================
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =============================================
# TRAIN
# =============================================
history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=val_data
)

# =============================================
# SAVE MODEL
# =============================================
model.save(MODEL_SAVE_PATH)
print(f"\n✅ Model saved to: {MODEL_SAVE_PATH}")
print(f"Training Accuracy: {max(history.history['accuracy'])*100:.2f}%")
print(f"Validation Accuracy: {max(history.history['val_accuracy'])*100:.2f}%")