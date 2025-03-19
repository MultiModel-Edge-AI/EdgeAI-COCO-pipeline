import os
import numpy as np
from skimage import io, color, transform
from skimage.feature import hog
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

food_images = 'Food_Images'
test_images = 'Test'
classes = ['hot_dog','french_fries','hamburger','pizza']  

#training data
train_files  = []
train_labels = []

for class_name in classes:
    class_dir = os.path.join(food_images, class_name)
    if not os.path.isdir(class_dir):
        raise ValueError(f"Training folder not found: {class_dir}")

    for fname in os.listdir(class_dir):
        lower_name = fname.lower()
        if lower_name.endswith('.jpg') or lower_name.endswith('.png'):
            full_path = os.path.join(class_dir, fname)
            train_files.append(full_path)
            train_labels.append(class_name)

#test Data
test_files   = []
test_labels  = []

if not os.path.isdir(test_images):
    raise ValueError(f"Test folder not found: {test_images}")

all_test_fnames = os.listdir(test_images)
for fname in all_test_fnames:
    lower_name = fname.lower()

    matched_class = None
    for class_name in classes:
        if class_name in lower_name:
            matched_class = class_name
            break

    if matched_class is None:
        print(f"Warning: Could not find class for test image: {fname}")
        continue

    test_files.append(os.path.join(test_images, fname))
    test_labels.append(matched_class)

#HOG
def extract_features(image_path):
    img = io.imread(image_path)
    if img.ndim == 3:
        img = color.rgb2gray(img)
    img = transform.resize(img, (128,128))

    hog_vec = hog(img,
                  orientations=9,
                  pixels_per_cell=(8,8),
                  cells_per_block=(2,2),
                  block_norm='L2-Hys',
                  transform_sqrt=True)
    return hog_vec

#feature extraction for training
X_train = []
for fpath in train_files:
    feats = extract_features(fpath)
    X_train.append(feats)

X_train = np.array(X_train, dtype=np.float32)

le = LabelEncoder()
y_train = le.fit_transform(train_labels)

#training decision tree
clf = DecisionTreeClassifier(min_samples_leaf=15)
clf.fit(X_train, y_train)

#test set evaluation
X_test = []
for fpath in test_files:
    feats = extract_features(fpath)
    X_test.append(feats)

X_test = np.array(X_test, dtype=np.float32)
y_test = le.transform(test_labels)

y_pred = clf.predict(X_test)

#confusion matrix for test set
cm_test = confusion_matrix(y_test, y_pred, labels=np.arange(len(le.classes_)))
print("Confusion Matrix (Test Set):\n", cm_test)

#error rates for test set
print("\nError Rates by Class (Test Set):")
num_classes = len(le.classes_)
row_sums = cm_test.sum(axis=1)

for i in range(num_classes):
    class_name = le.classes_[i]
    correct = cm_test[i, i]
    total   = row_sums[i]
    if total == 0:
        err_rate = 0.0
    else:
        err_rate = 1.0 - (correct / total)
    print(f"  {class_name}: {err_rate:.3f}")

#overall test error
total_correct = np.trace(cm_test)
total_images  = cm_test.sum()
test_error = 1.0 - (total_correct / total_images)
print(f"\nOverall Test Error: {test_error:.3f}")
print(f"Overall Test Accuracy: {1 - test_error:.3f}")

#evaluation on whole data set
all_files  = train_files + test_files
all_labels = train_labels + test_labels

X_all = []
for fpath in all_files:
    feats = extract_features(fpath)
    X_all.append(feats)

X_all = np.array(X_all, dtype=np.float32)
y_all = le.fit_transform(all_labels)

y_pred_all = clf.predict(X_all)

cm_all = confusion_matrix(y_all, y_pred_all, labels=np.arange(len(le.classes_)))
print("\nConfusion Matrix (All Data):\n", cm_all)

print("\nError Rates by Class (All Data):")
row_sums_all = cm_all.sum(axis=1)
for i in range(num_classes):
    class_name = le.classes_[i]
    correct = cm_all[i, i]
    total   = row_sums_all[i]
    if total == 0:
        err_rate = 0.0
    else:
        err_rate = 1.0 - (correct / total)
    print(f"  {class_name}: {err_rate:.3f}")

#overall error for whole dataset
total_correct_all = np.trace(cm_all)
total_images_all  = cm_all.sum()
all_error = 1.0 - (total_correct_all / total_images_all)
print(f"\nOverall Error (Train+Test): {all_error:.3f}")
print(f"Overall Accuracy (Train+Test): {1 - all_error:.3f}")
