import os
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# =================CONFIGURATION=================
# Folder names (Make sure these match your screenshots)
INPUT_ROOT = "Primary dataset"
OUTPUT_ROOT = "Augmented_Dataset"

# How many new images to create per ONE original image
AUGMENT_FACTOR = 30 
# ===============================================

# Define the Augmentation Rules (The "Photoshop" filters)
datagen = ImageDataGenerator(
    rotation_range=20,      # Rotate image slightly
    width_shift_range=0.1,  # Shift left/right
    height_shift_range=0.1, # Shift up/down
    shear_range=0.1,        # Slant the image
    zoom_range=0.2,         # Zoom in/out
    horizontal_flip=True,   # Mirror image
    brightness_range=[0.8, 1.2], # Darker/Brighter
    fill_mode='nearest'     # Fill empty space with nearest pixels
)

def augment_folder(class_name):
    # 1. Setup paths
    input_path = os.path.join(INPUT_ROOT, class_name)
    output_path = os.path.join(OUTPUT_ROOT, class_name)
    
    # 2. Create output directory (wipe it if it exists to start fresh)
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)
    
    print(f"Processing class: {class_name}...")
    
    # 3. Loop through every image in the folder
    files = [f for f in os.listdir(input_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for filename in files:
        # Load the image
        img = load_img(os.path.join(input_path, filename))
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape) # Reshape to (1, height, width, channels)
        
        # Generate variations
        i = 0
        for batch in datagen.flow(x, batch_size=1, 
                                  save_to_dir=output_path, 
                                  save_prefix=f"aug_{filename[:-4]}", 
                                  save_format='jpg'):
            i += 1
            if i >= AUGMENT_FACTOR:
                break 

    print(f"Finished {class_name}. Created ~{len(files) * AUGMENT_FACTOR} images.")

# Run for both folders
if not os.path.exists(INPUT_ROOT):
    print(f"Error: Could not find folder '{INPUT_ROOT}'. Check your spelling!")
else:
    augment_folder("potholes")
    augment_folder("roads")
    print(f"\nSUCCESS! Your new dataset is in '{OUTPUT_ROOT}'.")
