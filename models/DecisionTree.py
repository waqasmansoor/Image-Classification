from sklearn.tree import plot_tree,DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

PCA_NCOMP=0.95

class DecisionTree:
    def __init__(self,pca,max_depth=None,max_leaves=None,criterion="gini"):
        self.tree_seed=32
        self.max_depth=max_depth
        self.max_leaf_nodes=max_leaves
        self.criterion=criterion
        self.include_pca=pca
        if self.include_pca:
            self.pca=_PCA(PCA_NCOMP,random_state=52)

    def model(self):
        if self.include_pca:
            classifier = Pipeline([('pca',self.pca),
                        ('tree',DecisionTreeClassifier(criterion=self.criterion,max_depth=self.max_depth,
                                      max_leaf_nodes=self.max_leaf_nodes,random_state=self.tree_seed))])
        else:
            classifier= DecisionTreeClassifier(criterion=self.criterion,max_depth=self.max_depth,
                                      max_leaf_nodes=self.max_leaf_nodes,random_state=self.tree_seed)
        return classifier
    
    def predict(self,model,xtest,ytest):
        pred=model.predict(xtest)
        acc = accuracy_score(pred,ytest)
        return acc
    
    def modelInfo(self):
        pass

    

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