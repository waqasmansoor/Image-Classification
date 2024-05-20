from pathlib import  Path




if __name__ == '__main__':
    data_path=Path("C:/Users/PC/Desktop/Masters/Sem 4/COMP 6761/project/data")
    dir=['restaurant','library','lakeside','golf course','mosques']

    for d in dir:
        images_folder=Path(data_path/d)
        for img in images_folder.glob("*"):
            print(img)
            break