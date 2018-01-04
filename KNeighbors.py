from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#set data into input variables
X, y = make_blobs(centers=2, random_state=1234, cluster_std=0.3)

#Display x and y input variables
print("X- n_samples x n_features:", X.shape)
print("Y- n_samples:", y.shape)

#Display first 5 items from the datasets samples & lables
print("First 5 samples:", X[:5,:])
print("First 5 lables:", y[:5])

#spliting data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1234, stratify=y)

#display train and test data
print("X- train_data:", X_train)
print("X- test_data:", X_test)
print("y- trian_data:", y_train)
print("y- test_data:", y_test)

#Define the KNeighborsClassifier object
knn = KNeighborsClassifier(n_neighbors=3)

#trainning model
knn.fit(X_test, y_test)

#Display Accuracy of this model
print("---Displaying Accuracy of this model---")
accuracy = knn.score(X_test, y_test)

print("Accuracy:", accuracy)


