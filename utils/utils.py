import os
from pathlib import Path
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import random
import torch




class  imageUtils:
    def __init__(self,data_path,classes):
        self.data_path=data_path
        self.data_path=self.data_path.replace("/",os.sep)
        self.split_seed=47
        self.classes=classes


    def grayscale(self,x):
        gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    
        return gray

    def readImages(self,X,Y,img_size=64,gray=False,dataset="Train"):
        print(f"Reading {len(X)} of {dataset} dataset")
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
        for c in self.classes:
            d=Path(self.data_path+os.sep+c)
            for imp in d.glob("*"):
                self.images.append(imp)
                self.labels.append(self.class_one_hot[c])
        print(f"Loaded {len(self.images)} Image Paths to Ram")

    def oneHotEncoding(self):
        print(f"Encoding Images")
        self.class_one_hot={}
        try:
            for i,c in enumerate(self.classes):
                self.class_one_hot[c]=i
        except:
            print("Wrong Path")
        return self.class_one_hot
    
    def split(self,test_size,val_size,val=False,shuffle=True):
        if not val:
            print(f"Splitting Train and Test Dataset")
            x_train,x_test,y_train,y_test=train_test_split(self.images,self.labels,shuffle=shuffle,test_size=test_size,random_state=self.split_seed)
            return x_train,x_test,y_train,y_test    
        
        x_train,x_test,y_train,y_test=train_test_split(self.images,self.labels,shuffle=shuffle,test_size=test_size,random_state=self.split_seed)
        x_train,x_val,y_train,y_val=train_test_split(x_train,y_train,shuffle=shuffle,test_size=val_size,random_state=self.split_seed)
        return x_train,x_test,y_train,y_test,x_val,y_val
        
    
    


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
            

def seed_worker(worker_id):# class LoadImages(Dataset):
        
    """
    Sets the seed for a dataloader worker to ensure reproducibility, based on PyTorch's randomness notes.

    See https://pytorch.org/docs/stable/notes/randomness.html#dataloader.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)





def select_device():
    if torch.cuda.is_available():  # prefer GPU if available
        arg = "cuda:0"
    else:  # revert to CPU
        arg = "cpu"

    return torch.device(arg)

def model_info(model, verbose=False, imgsz=640):
    """
    Prints model summary including layers, parameters, gradients, and FLOPs; imgsz may be int or list.

    Example: img_size=640 or img_size=[640, 320]
    """
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        print(f"{'layer':>5} {'name':>40} {'gradient':>9} {'parameters':>12} {'shape':>20} {'mu':>10} {'sigma':>10}")
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace("module_list.", "")
            print(
                "%5g %40s %9s %12g %20s %10.3g %10.3g"
                % (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std())
            )

if __name__ == "__main__":
    data=Path("C:/Users/PC/Desktop/Masters/Sem 4/COMP 6761/project/data")
    classes=["Auditorium"]
    rename(classes,data)

    
            