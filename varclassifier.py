from sklearn.datasets import load_iris

irisds = load_iris()

A = irisds.data
B = irisds.target

from sklearn.cross_validation import train_test_split
A_train, A_test, B_train, B_test = train_test_split(A, B, test_size = .3)

from sklearn import tree
#own_classifier = tree.DecisionTreeClassifier()

from sklearn.neighbors import KNeighborsClassifier
own_classifier = KNeighborsClassifier(4)

own_classifier.fit(A_train, B_train)
predictions = own_classifier.predict(A_test)

from sklearn.metrics import accuracy_score
print (accuracy_score(B_test, predictions))
