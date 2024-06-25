import torch
from tqdm import tqdm
from pathlib import Path
import torch.nn as nn
import argparse
from utils.utils import imageUtils
from utils.dataloader import CreateDataloader

TQDM_BAR_FORMAT = "{l_bar}{bar:10}{r_bar}"  # tqdm bar format
batch_size=12
num_workers=1

def validate(model,dataloader,pbar,device,criterion,ret_result):
    training=model is not None
    if training:  # called by train.py
        device, pt, jit, engine = next(model.parameters()).device, True, False, False  # get model device, PyTorch model
        model.float()
    model.eval()
    pred, targets, loss= [], [], 0
    n = len(dataloader)  # number of batches
    action = "validating" if dataloader.dataset.name == "val" else "testing"
    desc = f"{pbar.desc[:-36]}{action:>36}" if pbar else f"{action}"
    bar = tqdm(dataloader, desc, n, not training, bar_format=TQDM_BAR_FORMAT, position=0)
#     with torch.cuda.amp.autocast(enabled=device.type != "cpu"):
    with torch.no_grad():
        for images, labels in bar:
            images, labels = images.to(device, non_blocking=True), labels.to(device)
            y = model(images)

            pred.append(y.argsort(1, descending=True)[:, :5])
            targets.append(labels)
            if criterion:
                loss += criterion(y, labels)

    loss /= n
    pred, targets = torch.cat(pred), torch.cat(targets)
    correct = (targets[:, None] == pred).float()
    if ret_result:
        return correct
    acc = torch.stack((correct[:, 0], correct.max(1).values), dim=1)  # (top1, top5) accuracy
    top1, top5 = acc.mean(0).tolist()

    if pbar:
        pbar.desc = f"{pbar.desc[:-36]}{loss:>12.3g}{top1:>12.3g}{top5:>12.3g}"
    

    return top1, top5, loss


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--test",default=False,help="Validate Test Set")
    
    opt = parser.parse_args()

    save_dir=Path("runs")
    weights=save_dir/"weights"/"best.pt"
    device="cuda"
    ckpt = torch.load(weights, map_location="cpu")  # load
    ckpt = ckpt["model"].to(device).float()  # FP32 model
    model=ckpt.eval()  # model in eval mode

    data_path="C:/Users/PC/Desktop/Masters/Sem 4/COMP 6761/project/data"
    classes=['restaurant','library']#,'lakeside','golfcourse','Auditorium']

    iu=imageUtils(data_path,classes)
    class_one_hot=iu.oneHotEncoding()
    print(f"Classes {class_one_hot}")
    iu.loadDataToRam()

    _,Xtest,_,Ytest,Xval,Yval=iu.split(0.3,0.2,val=True,shuffle=True)
    val_loader,_=CreateDataloader(Xval,Yval,None,batch_size,shuffle=True,workers=num_workers,seed=32,augment=False)
    test_loader,_=CreateDataloader(Xtest,Ytest,None,batch_size,shuffle=True,workers=num_workers,seed=32,augment=False)
    val_loader.dataset.name="Validation"
    test_loader.dataset.name="Test"

    criterion=nn.CrossEntropyLoss()
    if opt.test:
        dataloader=test_loader
    else:
        dataloader=val_loader
    device=model.device
    top1,_,loss = validate(
        model=model, dataloader=dataloader, criterion=criterion, pbar=None,device="cpu",ret_result=False
    )
    print(f"{dataloader.dataset.name}: Accuracy: {top1}, Loss: {loss.item()}")
    

