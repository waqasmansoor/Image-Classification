X_unlabeled, Y_unlabeled = loadImages(x_unlabeled, y_unlabeled)
X_unlabeled = np.array(X_unlabeled)
Y_unlabeled = np.array(Y_unlabeled)


test_pred = clf.predict(X_unlabeled)


accuracy = accuracy_score(Y_unlabeled, test_pred)
precision = precision_score(Y_unlabeled, test_pred, average='weighted')
recall = recall_score(Y_unlabeled, test_pred, average='weighted')
f1 = f1_score(Y_unlabeled, test_pred, average='weighted')


print("Final Metrics:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"F1 Score: {f1}")