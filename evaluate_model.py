import os
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg') # Prevents the crash
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ================= CONFIGURATION =================
MODEL_PATH = 'best_model.keras'
DATA_DIR = 'dataset' # Ensure this path is correct
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
# =================================================

def evaluate():
    print(f"Loading {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    print("\n--- Loading Validation Data (Correctly Mixed) ---")
    # CRITICAL FIX: shuffle=True to get the mixed set, not just one class
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True 
    )

    class_names = val_ds.class_names
    print(f"Classes: {class_names}")

    print("Running Predictions on mixed data...")
    y_true = []
    y_pred = []

    # Iterate through the batch to ensure images and labels stay paired
    for images, labels in val_ds:
        preds = model.predict(images, verbose=0)
        preds_binary = (preds > 0.5).astype(int).flatten()
        
        y_true.extend(labels.numpy())
        y_pred.extend(preds_binary)

    # CONFUSION MATRIX
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix_v2.png')
    print("\n[Saved] confusion_matrix_v2.png")

    # REPORT
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\n--- TRUE FINAL METRICS ---")
    print(report)
    
    with open('model_report_final.txt', 'w') as f:
        f.write(report)

if __name__ == "__main__":
    evaluate()