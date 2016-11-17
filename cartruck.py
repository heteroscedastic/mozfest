# Classify a Hatchback car and a Truck
from sklearn import tree
# 1 in feature shows its "long" and 0 shows its "short"
features = [[3000, 1], [5000, 1], [2000, 0], [1000, 0]]
# Tell scikit that 1 is a truck and 0 is a Hatchback car.
labels = [1, 1, 0, 0]
classify = tree.DecisionTreeClassifier()
classify = classify.fit(features, labels)
print (classify.predict([[1500, 0]]))
