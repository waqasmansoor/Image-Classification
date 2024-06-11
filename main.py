from utils.utils import imageUtils
import argparse
from models.DecisionTree import DecisionTree
from sklearn.model_selection import cross_validate,KFold
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from models.DecisiontreeSS import param_combinations

batch_size=8
num_workers=2


IMG_SIZE=64
MAX_DEPTH=[None,15,10,15,20,25]
MIN_SAMPLES_LEAF=[None,None,35,65,95,125]
MIN_SAMPLES_SPLIT=[None,None,200,210,240,300]
CRITERION="gini"

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='sup', help='sup/unsup')
    parser.add_argument("--pca",default=False,help="Implement PCA")
    parser.add_argument("--cv",default=False,help="Implement CrossValidation")
    parser.add_argument("--n",default=1,type=int,help="Training Iterations")
    parser.add_argument("--kf",default=5,type=int,help="KFold")
    
    opt = parser.parse_args()


    data_path="C:/Users/PC/Desktop/Masters/Sem 4/COMP 6761/project/data"
    classes=['restaurant','library']#,'lakeside','golfcourse','Auditorium']

    iu=imageUtils(data_path,classes)
    class_one_hot=iu.oneHotEncoding()
    print(f"Classes {class_one_hot}")
    iu.loadDataToRam()
    

    if opt.model == "sup":
        x_train,x_test,y_train,y_test=iu.split(0.3,0,val=False,shuffle=True)
        
        
        D,L=iu.readImages(x_train,y_train,IMG_SIZE,gray=False,dataset="Train")
        Dt,Lt=iu.readImages(x_test,y_test,IMG_SIZE,gray=False,dataset="Test")
        scoring = {
            'precision': make_scorer(precision_score, average='weighted'),
            'recall': make_scorer(recall_score, average='weighted'),
            'f1_score': make_scorer(f1_score, average='weighted')
        }
        results={}
        for i in range(opt.n):
            print(f"{i} -------------------")
            DT=DecisionTree(opt.pca,MAX_DEPTH[i],MIN_SAMPLES_LEAF[i],MIN_SAMPLES_SPLIT[i],CRITERION)
            dt=DT.model()
            if opt.cv:
                print(f"Running CrossValidation==> KFold: {opt.kf}")
                kf=KFold(n_splits=opt.kf)
                results[i]=cross_validate(dt,D,L,cv=kf,n_jobs=-1,scoring=scoring,return_train_score=True)
                print(f"Iter:{i}=> Test Precision: {results[i]['test_precision']}, Test Recall {results[i]['test_recall']}, Test F1: {results[i]['test_f1_score']}")
                print(f"Iter:{i}=> Train Precision: {results[i]['train_precision']}, Test Recall {results[i]['train_recall']}, Test F1: {results[i]['train_f1_score']}")
            else:
                print(f"Training Decision Tree==> Max Depth: {MAX_DEPTH[i]}, Min Leaves {MIN_SAMPLES_LEAF[i]}, Min Nodes {MIN_SAMPLES_SPLIT[i]}")
                classifier=dt.fit(D,L)
                nn,_=DT.modelInfo(dt)#Get number of nodes and max-depth
                n_leaves=DT.numOfLeaves(dt,nn)
                print(f"Number of Leaves {n_leaves}")
                print("Predicting...")
                result=DT.predict(classifier,Dt,Lt)
                print(f"Accuracy: {result}, Error: {1-result}")
    elif opt.model == "ss":
        Xlabeled,x_test,Ylabeled,y_test,Xunlabeled,Yunlabeled=iu.split(0.3,0.7,val=False,shuffle=True)
        
        D,L=iu.readImages(Xlabeled,Ylabeled,IMG_SIZE,gray=False,dataset="Train")
        Dv,Lv=iu.readImages(Xunlabeled,Yunlabeled,IMG_SIZE,gray=False,dataset="Validation")
        Dt,Lt=iu.readImages(x_test,y_test,IMG_SIZE,gray=False,dataset="Test")

        i=0
        while True:
            
            print(f"{i} -------------------")
            maxDepth=param_combinations[i]
            minSamplesLeaf=param_combinations[i]
            minSamplesSplit=param_combinations[i]
            DT=DecisionTree(opt.pca,maxDepth,minSamplesLeaf,minSamplesSplit,CRITERION)

            


    

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