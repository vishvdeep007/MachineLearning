from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

#fit data into variables
X, y = make_blobs(centers=2, random_state=0, cluster_std=0.3)

#print x and y
print("X- n_samples x n_features:", X.shape)
print("Y- n_samples:", y.shape)


#print first 5 data of the datasets
print("First 5 samples:", X[:5, :])
print("First 5 features:",y[:5])

#spliting data into trian and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1234, stratify=y)

#Display train and test data
print("X_train:", X_train)
print("X_test:", X_test)
print("y_train", y_train)
print("y_test:", y_test)

#define LogisticRegression object
classifier = LogisticRegression()

#trainning this model
classifier.fit(X_train, y_train)

prediction = classifier.predict(X_test)

print("---Comaparing the predition with test-data---")
print("prediction:", prediction)
print("y_test:", y_test)

#finding accuracy
accuracy = np.mean(prediction == y_test)
print("Accuracy of this model is:", accuracy)

#finding accuracy using lib function
accuracy = classifier.score(X_test, y_test)
print("Accuracy of this model usng lib function:", accuracy)

#print coef_ for input(x) and intercept_ for Bita
print(classifier.coef_)
print(classifier.intercept_)





