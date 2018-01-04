from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#import matplotlib.pyplot as plt

iris = load_iris()
X, y = iris.data, iris.target

#Display iris data and target
print("iris_data:", iris.data)
print("iris_target:", iris.target)

#spliting data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

#Display train and test data
print("X_train:", X_train)
print("X_test:", X_test)
print("y_train:", y_train)
print("y_test:", y_test)

classifier = LogisticRegression()

#trainning model
classifier.fit(X_test, y_test)

#Display Accuracy of this model
accuracy = classifier.score(X_test, y_test)
print("Accuracy of this model:", accuracy)

#Display graph
#plt.plot(iris.target, accuracy)
#plt.xlabel('iris_target')
#plt.ylabel('accuracy')
#plt.show()


