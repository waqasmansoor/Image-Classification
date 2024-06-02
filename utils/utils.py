import os
from pathlib import Path





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

    
            