import os
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import tree
import math
from sklearn.model_selection import cross_validate,KFold
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
import cv2
import numpy as np
from sklearn.metrics import accuracy_score


data_path=Path("/kaggle/input/venues-dataset-for-classification")
for classes in data_path.glob("*"):
    count=len(list(classes.glob("*")))
    print(f"Class {classes.name}, Number of Images {count}")
    plt.bar(classes.name,count)
plt.xlabel("Classes")
plt.ylabel("Number of Images")
plt.title("Dataset")
plt.show()

def loadImages(X,Y,img_size=64):
    D,L=[],[]
    for ix,iy in zip(X,Y):
        im=cv2.imread(str(ix))
        assert im is not None, f"Image Not Found {f}"
        h0, w0 = im.shape[:2]  # orig hw
        r = h0 != img_size or w0 != img_size  # ratio
        if r != 1:  # if sizes are not equal
            interp = cv2.INTER_AREA
            im=cv2.resize(im,(img_size,img_size),interpolation=interp)
        img = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img=img.ravel(order='K')
        D.append(img)
        L.append(iy)
    
    return D,L


class_one_hot={}
for i,classes in enumerate(data_path.glob("*")):
    class_one_hot[classes.name]=i
class_one_hot

images,labels=[],[]
for classes in data_path.glob("*"):
    for c in classes.glob("*"):
        images.append(c)
        labels.append(class_one_hot[classes.name])
print(f"Total Images: {len(images)}")

x_train,x_test,y_train,y_test=train_test_split(images,labels,shuffle=True,test_size=0.3,
                                              random_state=47)

print(f"Train: {len(x_train)}, Test: {len(x_test)}")

X,Y=loadImages(x_train,y_train)
X=np.array(X)
Y=np.array(Y)
X.shape

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,Y)

n_nodes=clf.tree_.node_count
max_depth=clf.tree_.max_depth

print(f"Nodes: {n_nodes}, Max Depth: {max_depth}")

Xtest,Ytest=loadImages(x_test,y_test)

Xtest=np.array(Xtest)
Ytest=np.array(Ytest)

pred=clf.predict(Xtest)

error_rate = 1 - accuracy_score(pred,Ytest)
print('Error rate',error_rate)