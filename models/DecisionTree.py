from sklearn.tree import plot_tree,DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import numpy as np

PCA_NCOMP=0.95


class DecisionTree:
    def __init__(self,pca,max_depth=None,min_samples_leaf=None,min_samples_split=None,criterion="gini"):
        self.tree_seed=32
        self.max_depth=max_depth
        self.min_samples_leaf=1 if min_samples_leaf == None else min_samples_leaf
        self.min_samples_split=2 if min_samples_split == None else min_samples_split
        self.criterion=criterion
        self.include_pca=pca
        if self.include_pca:
            self.pca=_PCA(PCA_NCOMP,random_state=52)

    def model(self):
        if self.include_pca:
            
            classifier = Pipeline([('pca',self.pca),
                        ('tree',DecisionTreeClassifier(criterion=self.criterion,max_depth=self.max_depth,
                                    min_samples_leaf=self.min_samples_leaf,
                                    min_samples_split=self.min_samples_split,random_state=self.tree_seed))])
        else:
            classifier= DecisionTreeClassifier(criterion=self.criterion,max_depth=self.max_depth,
                                    min_samples_leaf=self.min_samples_leaf,
                                    min_samples_split=self.min_samples_split,random_state=self.tree_seed)
        return classifier
    
    def predict(self,model,xtest,ytest):
        pred=model.predict(xtest)
        acc = accuracy_score(pred,ytest)
        return acc
    
    def modelInfo(self,clsfr):
        if len(clsfr)==2:
            n_nodes=clsfr[1].tree_.node_count
            max_depth=clsfr[1].tree_.max_depth
        else:
            n_nodes=clsfr.tree_.node_count
            max_depth=clsfr.tree_.max_depth
        print(f"Nodes: {n_nodes}, Max Depth: {max_depth}")
        return n_nodes,max_depth

    def numOfLeaves(self,clsfr,n_nodes):
        if len(clsfr)==2:
            clsfr=clsfr[1]
        child_left=clsfr.tree_.children_left
        child_right=clsfr.tree_.children_right

        is_leaves=np.zeros((n_nodes,),dtype=bool)
        stack=[(0,-1)]
        while len(stack) != 0:
            node_id,parent_depth=stack.pop()

        if child_left[node_id] != child_right[node_id]:
            stack.append((child_left[node_id],parent_depth+1))
            stack.append((child_right[node_id],parent_depth+1))
        else:
            is_leaves[node_id]=True

        n_leaves=0
        for i in range(len(is_leaves)):
            if is_leaves[i]:
                n_leaves+=1
        return n_leaves
        

    

class _PCA():
    def __init__(self, n_components, svd_solver='auto', compress=False, random_state=0):
        super().__init__()
        self.n_components = n_components
        self.svd_solver = svd_solver
        self.n_components_ = 0
        self.random_state = random_state
        self.PCA = PCA(n_components=n_components, svd_solver=svd_solver, random_state=random_state)
        self.compress = compress
        self.original_columns = []
        
    
    def fit(self, X, y=None):
        
        self.PCA.fit(X)
        self.n_components_ = self.PCA.n_components_
        
        return self
    
    def transform(self, X, y=None):
                
        X_tr = self.PCA.transform(X)
        
        return X_tr
    
    
#     def inverse_transform(self, X, y=None):
        
#         try:
#             X_tr = self.PCA.inverse_transform(X)
#         except ValueError:  # if self.compress=True, the inverse_transform called externally would throw an error
#             return X
#         X_tr = pd.DataFrame(X_tr, columns=self.original_columns)
        
#         return X_tr