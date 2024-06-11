import numpy as np
import math


param_combinations = [
    {'max_depth': 5, 'min_samples_split': 2, 'min_samples_leaf': 1},
    {'max_depth': 6, 'min_samples_split': 2, 'min_samples_leaf': 1},
    {'max_depth': 7, 'min_samples_split': 2, 'min_samples_leaf': 1},
    {'max_depth': 8, 'min_samples_split': 2, 'min_samples_leaf': 1},
    {'max_depth': 8, 'min_samples_split': 20, 'min_samples_leaf': 1},
    {'max_depth': 8, 'min_samples_split': 40, 'min_samples_leaf': 1},
    {'max_depth': 8, 'min_samples_split': 25, 'min_samples_leaf':10},
    {'max_depth': 8, 'min_samples_split': 25, 'min_samples_leaf': 30},
    {'max_depth': 8, 'min_samples_split': 25, 'min_samples_leaf': 50},
]

def predict_with_confidence(model, test,top10, threshold=0.9):
    probas = model.predict_proba(test)
    max_probas = np.max(probas, axis=1)
    confident_indices = max_probas >= threshold
    #print(sum(confident_indices))
    total_correct=sum(confident_indices)
#     top10=math.floor(len(X_unlabeled)*0.1)
    count=0
    
    top10_confident_indices=np.zeros(confident_indices.shape,dtype="bool")
    increase_complexity=False
    if total_correct > top10:
        increase_complexity=True
        for i,ci in enumerate(confident_indices):
            if ci:#true
                top10_confident_indices[i]=True
                count+=1
            if count == top10:
                break
        top10_confident_preds = np.argmax(probas[top10_confident_indices], axis=1)
        return top10_confident_indices,top10_confident_preds,increase_complexity
    else:    
        confident_preds = np.argmax(probas[confident_indices], axis=1)
    
        return confident_indices, confident_preds,increase_complexity
#     return confident_indices, confident_preds


def semiSupervised():


    top10=math.floor((len(X_labeled)+len(X_unlabeled))*0.1)
    train_size={}
    complexity=5
    i=8
    niter=0
    images_added=0
    while True:
        niter+=1
        
        clf=Pipeline([('pca',Clsfr_PCA(0.95,svd_solver="auto",compress=False,random_state=32)),
                    ("tree",DecisionTreeClassifier(**param_combinations[i]))])
        clf=clf.fit(X_labeled,Y_labeled)
        
        pred=clf.predict(X_unlabeled)
        acc = accuracy_score(pred,Y_unlabeled)
        print(f"Validation Accuracy {acc}")
        ts=len(X_labeled)
        if not ts in train_size:
            train_size[ts]={'test':0,'val':acc}
        
        
        confident_indices, confident_preds,c = predict_with_confidence(clf, X_unlabeled,top10)
        if len(confident_preds) < 10:
            print(f"Poor Accuracy, TP {confident_preds}")
            break 
    #     if c:
    #         print(f"Max Depth {complexity}")
    #         complexity+=1
        if not any(confident_indices):
            print('Not confident')
            break 
        
        print(f"Adding {len(confident_preds)} images in {len(X_labeled)} train dataset", sep=" ")
        images_added+=len(confident_preds)
        
        X_labeled = np.concatenate((X_labeled, X_unlabeled[confident_indices]))
        Y_labeled = np.concatenate((Y_labeled, confident_preds))
        
        
    #     print(f"Remaining Dataset in Labeled set {len(X_unlabeled)}")
        pred=clf.predict(Xtest)
        acc = accuracy_score(pred,Ytest)
        print(f"Test Accuracy {acc}, Unlabeled Data {len(X_unlabeled)}")
        train_size[ts]['test']=acc
        
        X_unlabeled = [x for i, x in enumerate(X_unlabeled) if not confident_indices[i]]
        Y_unlabeled = [y for i, y in enumerate(Y_unlabeled) if not confident_indices[i]]
        X_unlabeled=np.array(X_unlabeled)
        Y_unlabeled=np.array(Y_unlabeled)
        print(f"Number of Iterations {niter}, Total Images Added {images_added}")


    pred=clf.predict(Xtest)
    acc = accuracy_score(pred,Ytest)
    print(f"Accuracy {acc}, Unlabeled Data {len(X_unlabeled)}")
    ts=len(X_labeled)
    train_size[ts]['test']=acc    
