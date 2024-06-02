import os
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from utils.dataloader import CreateDataloader

batch_size=8
num_workers=2
seed=47

if __name__ == '__main__':
    data_path=Path("C:/Users/PC/Desktop/Masters/Sem 4/COMP 6761/project/data")
    dir=['restaurant','library','lakeside','golf course','mosques']

    class_one_hot={}
    for i,classes in enumerate(data_path.glob("*")):
        class_one_hot[classes.name]=i
    print("Labels ",class_one_hot)

    images,labels=[],[]
    for classes in data_path.glob("*"):
        for c in classes.glob("*"):
            images.append(c)
            labels.append(class_one_hot[classes.name])
    print(f"Total Images: {len(images)}")
    
    train_loader,dataset=CreateDataloader(images,labels,batch_size,shuffle=True,workers=num_workers,seed=seed)
    start_epoch=0
    epochs=1
    nb=len(train_loader)
    for epoch in range(start_epoch,epochs):
        for i, (imgs,target) in enumerate(train_loader):
            for im,l in zip(imgs,target):
                plt.imshow(im)
                print(l)
                plt.show()
            break