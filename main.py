import os
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from utils.dataloader import CreateDataloader
from utils.utils import imageUtils
import argparse
from models.DecisionTree import DecisionTree

batch_size=8
num_workers=2


IMG_SIZE=64

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='sup', help='sup/unsup')
    parser.add_argument("--pca",default=False,help="Implement PCA")
    
    opt = parser.parse_args()


    data_path="C:/Users/PC/Desktop/Masters/Sem 4/COMP 6761/project/data"
    classes=['restaurant','library']#,'lakeside','golfcourse','Auditorium']

    iu=imageUtils(data_path,classes)
    class_one_hot=iu.oneHotEncoding()
    print(f"Classes {class_one_hot}")
    iu.loadDataToRam()
    

    x_train,x_test,y_train,y_test=iu.split(0.3,0,val=False,shuffle=True)
    if opt.model == "sup":
        D,L=iu.readImages(x_train,y_train,IMG_SIZE,gray=False,dataset="Train")
        Dt,Lt=iu.readImages(x_test,y_test,IMG_SIZE,gray=False,dataset="Test")
        DT=DecisionTree(opt.pca)
        dt=DT.model()
        print(f"Training Decision Tree")
        classifier=dt.fit(D,L)
        print("Predicting...")
        result=DT.predict(classifier,Dt,Lt)
        print(f"Accuracy: {result}, Error: {1-result}")


    

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