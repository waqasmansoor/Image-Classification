from utils.utils import imageUtils,select_device,model_info
import argparse
from models.DecisionTree import DecisionTree
from sklearn.model_selection import cross_validate,KFold
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from models.DecisiontreeSS import param_combinations
import math
from models.DecisiontreeSS import semiSupervised
import yaml
from utils.dataloader import CreateDataloader
from models.cnn import Model
import torch
from utils.optimizer import smart_optimizer
from torch.cuda import amp
import time
from pathlib import Path
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from models.validate import validate
from copy import deepcopy
from utils.graphs import learning_curve

batch_size=8
num_workers=1


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
    parser.add_argument("--augment",default=False,help="Add Augmentatino in CNN Training")
    
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
        Xlabeled,x_test,Ylabeled,y_test,Xunlabeled,Yunlabeled=iu.split(0.3,0.7,val=True,shuffle=True)
        
    
        for i in range(len(param_combinations)):
            D,L=iu.readImages(Xlabeled,Ylabeled,IMG_SIZE,gray=False,dataset="Train")
            Dv,Lv=iu.readImages(Xunlabeled,Yunlabeled,IMG_SIZE,gray=False,dataset="Validation")
            Dt,Lt=iu.readImages(x_test,y_test,IMG_SIZE,gray=False,dataset="Test")
            top10=math.floor((len(D)+len(Dv))*0.1)
            
            maxDepth=param_combinations[i]['max_depth']
            minSamplesLeaf=param_combinations[i]['min_samples_leaf']
            minSamplesSplit=param_combinations[i]['min_samples_split']
            print(f"-----> {i}: Max Depth: {maxDepth}, Min Leaves {minSamplesLeaf}, Min Nodes {minSamplesSplit}, Number of Images to Add: {top10}")
            model=DecisionTree(opt.pca,maxDepth,minSamplesLeaf,minSamplesSplit,CRITERION)
            semiSupervised(model,D,L,Dv,Lv,Dt,Lt,top10)
            print()
            
    elif opt.model == "cnn":

        Xtrain,Xtest,Ytrain,Ytest,Xval,Yval=iu.split(0.3,0.2,val=True,shuffle=True)
        hypp="config/augment.yaml"
        with open(hypp, encoding="ascii", errors="ignore") as f:
            hyp = yaml.safe_load(f)  # model dict


        seed=47
        nc=len(class_one_hot)

        train_loader,dataset=CreateDataloader(Xtrain,Ytrain,hyp,batch_size,shuffle=True,workers=num_workers,seed=seed,augment=opt.augment)

        train_loader.dataset.name="train"
        val_loader,_=CreateDataloader(Xval,Yval,hyp,batch_size,shuffle=True,workers=num_workers,seed=seed,augment=False)
        test_loader,_=CreateDataloader(Xtest,Ytest,hyp,batch_size,shuffle=True,workers=num_workers,seed=seed,augment=False)
        val_loader.dataset.name="val"
        test_loader.dataset.name="test"

        device=select_device()
        model_cfg="models/cnn.yaml"
        model=Model(model_cfg,nc=nc,ch=3)
        dropout=0.7
        for m in model.modules():
            if hasattr(m,'reset_parameters'):
                m.reset_parameters()
            if isinstance(m,torch.nn.Dropout) and dropout is not None:
                m.p=dropout
        for p in model.parameters():
            p.requires_grad=True
        model=model.to(device)
        model.device=device
    
        model.names=list(class_one_hot.keys())
        model.transforms=test_loader.dataset.preprocess
        model_info(model,True)


        optimizer=["SGD", "Adam", "AdamW", "RMSProp"]
        lr0=0.001
        lrf=0.01
        decay=5e-5
        epochs=50
        lf = lambda x: (1 - x / epochs) * (1 - lrf) + lrf  # linear

        optimizer = smart_optimizer(model, optimizer[1], lr0, momentum=0.9, decay=decay)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
        criterion=nn.CrossEntropyLoss()
        TQDM_BAR_FORMAT = "{l_bar}{bar:10}{r_bar}"  # tqdm bar format

        start_epoch=0
        cuda = device.type != "cpu"

        nb=len(train_loader)
        scaler = amp.GradScaler(enabled=cuda)
        t0 = time.time()
        val="val"
        best_fitness=0.0
        save_dir=Path("runs")
        wdir = save_dir / "weights"
        wdir.mkdir(parents=True, exist_ok=True)  # make dir
        best = wdir / "best.pt"
        train_loss,val_loss=[],[]
        for epoch in range(start_epoch,epochs):
            tloss, vloss, fitness = 0.0, 0.0, 0.0 
            model.train()
            pbar = tqdm(enumerate(train_loader), total=len(train_loader), bar_format=TQDM_BAR_FORMAT)
            for i, (imgs,target) in pbar:
                images,labels=imgs.to(device,non_blocking=True),target.to(device)
                with amp.autocast(enabled=cuda):  # stability issues when enabled
                    loss = criterion(model(images), labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                tloss = (tloss * i + loss.item()) / (i + 1)  # update mean losses
                mem = "%.3gG" % (torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0)  # (GB)
                pbar.desc = f"{f'{epoch + 1}/{epochs}':>10}{mem:>10}{tloss:>12.3g}" + " " * 36
                
                if i == len(pbar) - 1:  # last batch
                    top1, top5, vloss = validate(
                        model=model, dataloader=val_loader, criterion=criterion, pbar=pbar,device='cuda',ret_result=False
                    )  # test accuracy, loss
                    fitness = top1  # define fitness as top1 accuracy
                    train_loss.append(tloss)
                    val_loss.append(vloss.item())
            scheduler.step()
            if fitness > best_fitness:
                best_fitness = fitness
            metrics = {
                "train/loss": tloss,
                f"{val}/loss": vloss.item(),
                "metrics/accuracy_top1": top1,
                "metrics/accuracy_top5": top5,
                "lr/0": optimizer.param_groups[0]["lr"],
                }  # learning rate
            print(metrics)
            
            ckpt = {
                "epoch": epoch,
                "best_fitness": best_fitness,
                "model": deepcopy(model).half(),  # deepcopy(de_parallel(model)).half(),
                "optimizer": None,  # optimizer.state_dict(),
            }

            # Save last, best and delete
            if best_fitness == fitness:
                torch.save(ckpt, best)
            del ckpt
        
        #Training Finished
        learning_curve(train_loss,val_loss)


