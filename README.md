# Image Classification with CNN and Supervised and Semi-Supervised Decision Trees

## Installations
Requirements
``` shell
pip install -r requirements.txt
```


## Decisiont Tree
Run Decision Tree with Default Parameters
``` shell
python main.py --model sup
```

## Decision Tree with PCA
Run Decison Tree with PCA
``` shell
python main.py --model sup --pca True
```
## Hyperparameter Configuration
You can run Decision Tree training with different parameters to avoid overfitting. The following parameters can be changes
- Max Depth
- Min Leaves
- Min Nodes

To change hyperparameters, change the values of the respective variables in the <b>main.py</b> file. Run the following code for the required number of iterations.
``` shell
python main.py --model sup --pca True --n 1
```

## CrossValidating Hyperparameters
Run cross validation to see performance on the validation dataset. 
The default number of <b>KFolds</b> are 5
``` shell
python main.py --model sup --pca True --kf 5 --cv True
```

## Semi-supervised Learning with Decision Tree
Semi-supervised learning with PCA can be run by the following code
``` shell
python main.py --model ss --pca True
```
The Dataset ratio for Train:Val:Test is 30:70:30.
We use the following hyperparamters to analyse Semi-supervised learning with Decision Tree.
- Max Depth [5,6,7,8]
- Min Leaves [2,2,25,40]
- Min Nodes [1,10,50]

## CNN
Run Convolutional Neural Network on the Train Dataset
```shell
python main.py --model cnn
```
### Inference
We can predict the test and val dataset on the best trained model by using the following code
```shell
python validate.py
```

