import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg') # Silent mode
import matplotlib.pyplot as plt

# ================= CONFIGURATION =================
MODEL_PATH = 'best_model.keras'
IMG_SIZE = (224, 224)
THRESHOLD = 0.5  # < 0.5 is Normal, > 0.5 is Pothole
# =================================================

def predict_image(image_path):
    # 1. Load Model
    if not os.path.exists(MODEL_PATH):
        print("Error: Model not found.")
        return
    
    print(f"Loading Model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    # 2. Load & Preprocess Image
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found.")
        return

    print(f"Analyzing {image_path}...")
    img = cv2.imread(image_path)
    
    # Resize and Scale (Just like training)
    img_resized = cv2.resize(img, IMG_SIZE)
    img_array = np.expand_dims(img_resized, axis=0) # Shape: (1, 224, 224, 3)

    # 3. Predict
    prediction = model.predict(img_array, verbose=0)[0][0]
    
    # 4. Interpret Result
    confidence = prediction if prediction > 0.5 else 1 - prediction
    label = "POTHOLE" if prediction > THRESHOLD else "NORMAL ROAD"
    color = (0, 0, 255) if label == "POTHOLE" else (0, 255, 0) # Red or Green

    print("-" * 30)
    print(f"RESULT: {label}")
    print(f"Confidence: {confidence * 100:.2f}%")
    print(f"Raw Score: {prediction:.4f}")
    print("-" * 30)

    # 5. Save Visual Result
    # Draw text on image and save it
    cv2.putText(img, f"{label} ({confidence*100:.1f}%)", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    output_filename = f"prediction_{os.path.basename(image_path)}"
    cv2.imwrite(output_filename, img)
    print(f"Saved result to: {output_filename}")

if __name__ == "__main__":
    # ASK USER FOR IMAGE
    path = input("Enter the path to your image (e.g., test.jpg): ")
    # Remove quotes if user added them
    path = path.strip().replace("'", "").replace('"', "")
    predict_image(path)