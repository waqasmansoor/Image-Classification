import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import cv2
import numpy as np




class  imageUtils:
    def __init__(self,data_path):
        self.data_path=data_path
        self.split_seed=47


    def grayscale(self,x):
        gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    
        return gray

    def readImages(self,X,Y,img_size=64,gray=False):
        D,L=[],[]
        for ix,iy in zip(X,Y):
            im=cv2.imread(str(ix))
            if gray:
                im=self.grayscale(im)
            assert im is not None, f"Image Not Found {im}"
            h0, w0 = im.shape[:2]  # orig hw
            r = h0 != img_size or w0 != img_size  # ratio
            if r:  # if sizes are not equal
                interp = cv2.INTER_AREA
                im=cv2.resize(im,(img_size,img_size),interpolation=interp)
            if not gray:
                pass
    #             img = np.flip(im, 2)

    #         img = im.reshape(1,-1, 3)
    #         r=img[0,:,0]
    #         g=img[0,:,1]
    #         b=img[0,:,2]
            img=im.ravel(order='K')
    #         D.append(np.hstack((r,g,b)))
            D.append(img)
    #         D.append(b)
            L.append(iy)
        
        return np.array(D),np.array(L)

    def loadDataToRam(self):
        self.images,self.labels=[],[]
        for classes in self.data_path.glob("*"):
            for c in classes.glob("*"):
                self.images.append(c)
                self.labels.append(self.class_one_hot[classes.name])

    def oneHotEncoding(self):
        self.class_one_hot={}
        try:
            for i,classes in enumerate(self.data_path.glob("*")):
                self.class_one_hot[classes.name]=i
        except:
            print("Wrong Path")
        return self.class_one_hot
    
    def split(self,test_size,val_size,val=False,shuffle=True):
        x_train,x_test,y_train,y_test=train_test_split(self.images,self.labels,shuffle=shuffle,test_size=test_size,random_state=self.split_seed)
        if val:
            x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,shuffle=shuffle,test_size=val_size,random_state=self.split_seed)
            return x_train,x_test,y_train,y_test,x_val,y_val
        return x_train,x_test,y_train,y_test
    
    


def rename(classes:list,path:str)->None:
    for c in classes:
        images_folder=Path(path/c)
        print(images_folder)
        for i,img in enumerate(images_folder.glob("*")):
            suff=Path(img).suffix
            new_name=str(images_folder)+os.sep+c+str(i)+suff
    
            if os.path.exists(img):
                print(img)
                os.rename(img,new_name)
            else:
                print(f"{path} not exists")
            




if __name__ == "__main__":
    data=Path("C:/Users/PC/Desktop/Masters/Sem 4/COMP 6761/project/data")
    classes=["Auditorium"]
    rename(classes,data)

    
            