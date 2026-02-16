import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
from sklearn.utils import class_weight

# ================= CONFIGURATION =================
DATA_DIR = "dataset"  # Now inside project_folder
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 25  # We have EarlyStopping, so 25 is safe
# =================================================

def train_final_model():
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")

    # 1. LOAD DATA
    print("\n--- Loading Dataset ---")
    
    # Training Set (80%)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='binary' 
    )

    # Validation Set (20%)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='binary'
    )

    class_names = train_ds.class_names
    print(f"Detected Classes: {class_names}")
    # Note: Alphabetical order. likely ['normal', 'potholes']
    # So 0 = normal, 1 = potholes

    # 2. CALCULATE CLASS WEIGHTS (The Fix for Imbalance)
    print("\n--- Calculating Class Weights ---")
    
    # We need to count files manually to get exact weights
    # Assuming class_names[0] is 'normal' and [1] is 'potholes'
    count_0 = len(os.listdir(os.path.join(DATA_DIR, class_names[0])))
    count_1 = len(os.listdir(os.path.join(DATA_DIR, class_names[1])))
    total = count_0 + count_1

    print(f"Count {class_names[0]}: {count_0}")
    print(f"Count {class_names[1]}: {count_1}")

    # Formula: Total / (2 * Count)
    weight_0 = (1 / count_0) * (total / 2.0)
    weight_1 = (1 / count_1) * (total / 2.0)

    class_weights = {0: weight_0, 1: weight_1}
    print(f"Computed Weights: {class_weights}")
    print("(The smaller class gets a higher weight to balance training)")

    # 3. PERFORMANCE TUNING
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # 4. BUILD MODEL (Professional CNN Architecture)
    model = models.Sequential([
        # Input & Rescaling
        layers.Input(shape=(224, 224, 3)),
        layers.Rescaling(1./255),
        
        # Block 1
        layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Block 2
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Block 3
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        # Block 4 (Deep Features)
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Classifier
        layers.GlobalAveragePooling2D(), # Better than Flatten for 224x224
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5), # Prevent Overfitting
        layers.Dense(1, activation='sigmoid') # Binary Output
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # 5. CALLBACKS
    callbacks_list = [
        callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        callbacks.ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_accuracy')
    ]

    # 6. TRAIN
    print("\n--- Starting Training ---")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks_list,
        class_weight=class_weights # <--- Applying the fix here!
    )

    # 7. PLOT RESULTS
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Loss')
    plt.legend()
    plt.savefig('training_results.png') # Save plot for your report
    print("\nResults saved to 'training_results.png'")
    print("Best Model saved as 'best_model.keras'")

if __name__ == "__main__":
    train_final_model()