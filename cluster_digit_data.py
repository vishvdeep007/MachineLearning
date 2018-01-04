from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans


digits = load_digits()

X, y = digits.data, digits.target

#display data
print("---Dislay the data of the digits data and target---")
print("\nData:", X.data)
print("\nTarget:", y.target)

#display shape of the data and target
print("---Dislay the shape of the digits data and target---")
print("\nData:", X.shape)
print("\nTarget:", y.shape)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

#display the train and test data of the X and Y
print("\n---Display the trian and test data of X---")
print("\nTrain data of X:", X_trian)
print("\nTest data of X:", X_test)
print("\n---Display the train and test data of Y---")
print("\nTrain data of Y:", y_train)
print("\nTest data of Y:", y_test)

kmeans = KMeans(n_clusters=10, random_state=42)

labels = kmeans.fir_predict(X)

#display cluster
print("\n---Display The Cluster of Digits data---")
print("\nCluster:", labels)

print("success!")
