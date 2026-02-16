import os
import cv2
import numpy as np
import tensorflow as tf

# ================= CONFIGURATION =================
MODEL_PATH = 'best_model.keras'
TEST_FOLDER = 'test_images' # Put your 5 images here
IMG_SIZE = (224, 224)
# =================================================

def batch_predict():
    if not os.path.exists(TEST_FOLDER):
        os.makedirs(TEST_FOLDER)
        print(f"Created folder '{TEST_FOLDER}'. Put your images there and run again!")
        return

    print(f"Loading Model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    files = [f for f in os.listdir(TEST_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"\nFound {len(files)} images. Analyzing...\n")

    print(f"{'FILENAME':<30} | {'PREDICTION':<15} | {'CONFIDENCE'}")
    print("-" * 60)

    for file in files:
        path = os.path.join(TEST_FOLDER, file)
        img = cv2.imread(path)
        if img is None: continue
        
        # Preprocess
        img_resized = cv2.resize(img, IMG_SIZE)
        img_array = np.expand_dims(img_resized, axis=0)
        
        # Predict
        score = model.predict(img_array, verbose=0)[0][0]
        
        label = "POTHOLE" if score > 0.5 else "ROAD"
        confidence = score if score > 0.5 else 1 - score
        
        print(f"{file:<30} | {label:<15} | {confidence*100:.2f}%")

if __name__ == "__main__":
    batch_predict()