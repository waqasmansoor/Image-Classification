import os
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import math
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
import cv2
import numpy as np
from sklearn.metrics import accuracy_score


data_path = Path("/kaggle/input/venues-dataset-for-classification")

for classes in data_path.glob("*"):
    count = len(list(classes.glob("*")))
    print(f"Class {classes.name}, Number of Images {count}")
    plt.bar(classes.name, count)
plt.xlabel("Classes")
plt.ylabel("Number of Images")
plt.title("Dataset")
plt.show()


def loadImages(X, Y, img_size=64):
    D, L = [], []
    for ix, iy in zip(X, Y):
        im = cv2.imread(str(ix))
        assert im is not None, f"Image Not Found {ix}"
        h0, w0 = im.shape[:2] 
        r = h0 != img_size or w0 != img_size 
        if r:  # if sizes are not equal
            interp = cv2.INTER_AREA
            im = cv2.resize(im, (img_size, img_size), interpolation=interp)
        img = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = img.ravel(order='K')
        D.append(img)
        L.append(iy)
    
    return D, L


class_one_hot = {}
for i, classes in enumerate(data_path.glob("*")):
    class_one_hot[classes.name] = i

images, labels = [], []
for classes in data_path.glob("*"):
    for c in classes.glob("*"):
        images.append(c)
        labels.append(class_one_hot[classes.name])
print(f"Total Images: {len(images)}")


def predict_with_confidence(model, X, threshold=0.9):
    probas = model.predict_proba(X)
    max_probas = np.max(probas, axis=1)
    confident_indices = max_probas >= threshold
    confident_preds = np.argmax(probas[confident_indices], axis=1)
    return confident_indices, confident_preds


x_labeled, x_unlabeled, y_labeled, y_unlabeled = train_test_split(images, labels, test_size=0.8, random_state=47)
X_labeled, Y_labeled = loadImages(x_labeled, y_labeled)
X_labeled = np.array(X_labeled)
Y_labeled = np.array(Y_labeled)


print(f"Initial Labeled Size: {len(Y_labeled)}, Initial Unlabeled Size: {len(y_unlabeled)}")
while True:
    clf = DecisionTreeClassifier(max_depth=10, min_samples_split=10, min_samples_leaf=5, max_leaf_nodes=100)
    clf = clf.fit(X_labeled, Y_labeled)
    
    X_unlabeled, _ = loadImages(x_unlabeled, y_unlabeled)
    X_unlabeled = np.array(X_unlabeled)
    
    if X_unlabeled.shape[0] == 0:
        print("Empty")
        break 
    
    confident_indices, confident_preds = predict_with_confidence(clf, X_unlabeled)
    
    if not any(confident_indices):
        print('Not confident')
        break 
    
    X_labeled = np.concatenate((X_labeled, X_unlabeled[confident_indices]))
    Y_labeled = np.concatenate((Y_labeled, confident_preds))
    
    x_unlabeled = [x for i, x in enumerate(x_unlabeled) if not confident_indices[i]]
    y_unlabeled = [y for i, y in enumerate(y_unlabeled) if not confident_indices[i]]
    
    print(f"Added {sum(confident_indices)} pseudo-labeled samples")
    
