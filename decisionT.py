import numpy as np
from sklearn import tree
from sklearn.datasets import load_iris
irisds = load_iris()
print (irisds.feature_names)
print (irisds.target_names)
print (irisds.data[0])
print (irisds.target[0])


test_id = [0, 1, 50, 51, 100, 101]

train_target = np.delete(irisds.target, test_id)
train_data = np.delete(irisds.data, test_id, axis = 0)

test_target = irisds.target[test_id]
test_data = irisds.data[test_id]

clf_iris = tree.DecisionTreeClassifier()
clf_iris.fit(train_data, train_target)

print (test_target)
print (clf_iris.predict(test_data))
