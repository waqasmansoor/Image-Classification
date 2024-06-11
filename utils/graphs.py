import matplotlib.pyplot as plt
from pathlib import Path

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
        