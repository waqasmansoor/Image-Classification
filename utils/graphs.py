import matplotlib.pyplot as plt
from pathlib import Path


def bar():
    data_path=Path("C:/Users/PC/Desktop/Masters/Sem 4/COMP 6761/project/data")


    dir=['restaurant','library','lakeside','golf course','mosques']
    for p in dir:
        image_folder=Path(data_path/p)
        s=len(list(image_folder.glob("*")))
        
        plt.bar(p,s)

    plt.title("Number of Images in each class")
    plt.xlabel("Class Names")
    plt.ylabel("No of Images")
    plt.show()
        
def learning_curve(train_loss,val_loss):
    plt.title("Learning Curve")
    plt.plot(train_loss,label="Train")

    plt.plot(val_loss,label="Val")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()