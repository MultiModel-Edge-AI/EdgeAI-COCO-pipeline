import numpy as np
from skimage import io, color, transform
from skimage.feature import hog

def extract_hog_feature(image_path, output_size=(128, 128)):
    
    #Loads an image from disk, converts to grayscale, resizes, and extracts HOG features.
    img = io.imread(image_path)
    # Convert to grayscale if needed
    if img.ndim == 3:
        img = color.rgb2gray(img)
    # Resize
    img = transform.resize(img, output_size)
    # Extract HOG
    hog_vec = hog(img,
                  orientations=9,
                  pixels_per_cell=(8, 8),
                  cells_per_block=(2, 2),
                  block_norm='L2-Hys',
                  transform_sqrt=True)
    return hog_vec

def build_dataset(image_label_pairs, output_size=(128, 128)):

# Given a list of (image_path, label), extract HOG features for each and return (X, y) as NumPy arrays.
    X_list, y_list = [], []
    for (fpath, label) in image_label_pairs:
        feature_vec = extract_hog_feature(fpath, output_size=output_size)
        X_list.append(feature_vec)
        y_list.append(label)
    X_arr = np.array(X_list, dtype=np.float32)
    return X_arr, y_list