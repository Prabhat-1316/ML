'''exp-8: Write a program to implement k-Nearest Neighbour algorithm to classify the iris data set.
Print both correct and wrong predictions. Java/Python ML library classes can be used for
this problem.'''

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize k-NN classifier
knn = KNeighborsClassifier(n_neighbors=3)

# Train the classifier
knn.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print correct and wrong predictions
print("\nCorrect predictions:")
for i in range(len(y_test)):
    if y_pred[i] == y_test[i]:
        print("Predicted:", y_pred[i], "Actual:", y_test[i])

print("\nWrong predictions:")
for i in range(len(y_test)):
    if y_pred[i] != y_test[i]:
        print("Predicted:", y_pred[i], "Actual:", y_test[i])

'''OUTPUT
Accuracy: 1.0

Correct predictions:
Predicted: 1 Actual: 1
Predicted: 0 Actual: 0
Predicted: 2 Actual: 2
Predicted: 1 Actual: 1
Predicted: 1 Actual: 1
Predicted: 0 Actual: 0
Predicted: 1 Actual: 1
Predicted: 2 Actual: 2
Predicted: 1 Actual: 1
Predicted: 1 Actual: 1
Predicted: 2 Actual: 2
Predicted: 0 Actual: 0
Predicted: 0 Actual: 0
Predicted: 0 Actual: 0
Predicted: 0 Actual: 0
Predicted: 1 Actual: 1
Predicted: 2 Actual: 2
Predicted: 1 Actual: 1
Predicted: 1 Actual: 1
Predicted: 2 Actual: 2
Predicted: 0 Actual: 0
Predicted: 2 Actual: 2
Predicted: 0 Actual: 0
Predicted: 2 Actual: 2
Predicted: 2 Actual: 2
Predicted: 2 Actual: 2
Predicted: 2 Actual: 2
Predicted: 2 Actual: 2
Predicted: 0 Actual: 0
Predicted: 0 Actual: 0
Predicted: 0 Actual: 0
Predicted: 0 Actual: 0
Predicted: 1 Actual: 1
Predicted: 0 Actual: 0
Predicted: 0 Actual: 0
Predicted: 2 Actual: 2
Predicted: 1 Actual: 1
Predicted: 0 Actual: 0
Predicted: 0 Actual: 0
Predicted: 0 Actual: 0
Predicted: 2 Actual: 2
Predicted: 1 Actual: 1
Predicted: 1 Actual: 1
Predicted: 0 Actual: 0
Predicted: 0 Actual: 0

Wrong predictions:
'''