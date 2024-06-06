from sklearn.tree import plot_tree,DecisionTreeClassifier
from sklearn.metrics import accuracy_score


class DT:
    def __init__(self,max_depth=None,max_leaves=None,criterion="gini"):
        self.tree_seed=32
        self.max_depth=max_depth
        self.max_leaf_nodes=max_leaves
        self.criterion=criterion

    def model(self):
        classifier= DecisionTreeClassifier(criterion=self.criterion,max_depth=self.max_depth,
                                      max_leaf_nodes=self.max_leaf_nodes,random_state=self.tree_seed)
        return classifier
    
    def predict(self,model,xtest,ytest):
        pred=model.predict(xtest)
        acc = accuracy_score(pred,ytest)
        return acc
    
    def modelInfo(self):
        pass

    