import os
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from utils.dataloader import CreateDataloader
from utils import imageUtils
import argparse

batch_size=8
num_workers=2
seed=47

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='sup', help='sup/unsup')
    
    opt = parser.parse_args()


    data_path=Path("C:/Users/PC/Desktop/Masters/Sem 4/COMP 6761/project/data")
    dir=['restaurant','library','lakeside','golf course','Auditorium']

    iu=imageUtils(data_path)
    class_one_hot=iu.oneHotEncoder()
    print(f"Classes {class_one_hot}")
    iu.loadDataToRam()
    

    print(f"Total Images: {len(iu.images)}")

    x_train,x_test,y_train,y_test=iu.split(0.3,0,False,True)
    if opt.model == "sup":
        pass


    

    """
        CNN
    """
    # train_loader,dataset=CreateDataloader(images,labels,batch_size,shuffle=True,workers=num_workers,seed=seed)
    # start_epoch=0
    # epochs=1
    # nb=len(train_loader)
    # for epoch in range(start_epoch,epochs):
    #     for i, (imgs,target) in enumerate(train_loader):
    #         for im,l in zip(imgs,target):
    #             plt.imshow(im)
    #             print(l)
    #             plt.show()
    #         break