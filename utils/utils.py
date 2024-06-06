import os
from pathlib import Path
from sklearn.model_selection import train_test_split




class  imageUtils:
    def __init__(self,data_path):
        self.data_path=data_path
        self.split_seed=47

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

    
            