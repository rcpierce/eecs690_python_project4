'''
Ryan Pierce
ID: 2317826
EECS 690 - Intro to Machine Learning, Python Project 4
'''

# Load libraries
from math import exp
from math import floor
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import random
import numpy as np
from numpy import array
from numpy import mean
from numpy import cov
from numpy.linalg import eig
print ('Libraries loaded successfully!')

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)
print ('Dataset loaded successfully!')

# Array containing initial dataset
data_array = dataset.values

# For X: select all rows, but only columns indexed from 0 to 3 (inputs)
X = data_array[:, 0:4]

print("|-------- Part 2: PCA --------|")

# For X: select all rows, but only columns indexed from 0 to 3 (inputs)
X = data_array[:, 0:4]

pca = PCA(4)
pca.fit(X)

X_transformed = pca.transform(X)
# Since we only need the first feature we can make all rows equal to
# the first feature.
for row in X_transformed:
    row = row[0]

# For y: select all rows, but only the last column (outputs)
y = data_array[:,4]

# Splitting the data in half, then creating two folds.
X_train_fold1, X_test_fold1, y_train_fold1, y_test_fold1 = train_test_split(X_transformed, y, test_size = 0.50, random_state = 1)

X_train_fold2 = X_test_fold1
X_test_fold2 = X_train_fold1
y_train_fold2 = y_test_fold1
y_test_fold2 = y_train_fold1

decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train_fold1, y_train_fold1)
pred_f1 = decision_tree.predict(X_test_fold1)
decision_tree.fit(X_train_fold2, y_train_fold2)
pred_f2 = decision_tree.predict(X_test_fold2)

predictions = np.concatenate([pred_f1, pred_f2])
y_test = np.concatenate((y_test_fold1, y_test_fold2))

print("\nEigenvectors:")
print(pca.components_)
print("\nEigenvalues:")
print(pca.explained_variance_)

pov = pca.explained_variance_[0] / sum(pca.explained_variance_)
print("\nPoV:")
print(pov)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nAccuracy Score:")
print(accuracy_score(y_test, predictions))
print("\n\n")

print("|-------- Part 3: Simulated Annealing --------|")

# For X: select all rows, but only columns indexed from 0 to 3 (inputs)
X = data_array[:, 0:4]

# For y: select all rows, but only the last column (outputs)
y = data_array[:,4]
feature_names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'sepal-length-prime', 'sepal-width-prime', 'petal-length-prime', 'petal-width-prime']
combined_features_orig = np.column_stack((X, X_transformed))
current_feature_set = combined_features_orig
num_iterations = 100
for i in range(num_iterations):
    temp_arr = current_feature_set
    perturb_num =  floor(random.uniform(0.01, 0.06) * num_iterations)
    if random.randint(0, 100) % 2 == 0:
        for j in range(perturb_num):
            concat_array = []
            concat_array.append(combined_features_orig[:, random.randint(0, 7)])
            concat_array = np.array(concat_array)
            concat_array = concat_array.reshape((150, 1))
            temp_arr = np.append(temp_arr, concat_array, 1)
    else:
        for j in range(perturb_num):
            rand_num = 0
            if np.size(temp_arr, 1) == 0:
                rand_num = 0
            else:
                rand_num = random.randint(0, np.size(temp_arr, 1)) - 1
            temp_arr = np.delete(temp_arr, rand_num, 1)
        
    perturbed_feature_set = temp_arr
    # Calculating accuracy for original set
    X_train_fold1, X_test_fold1, y_train_fold1, y_test_fold1 = train_test_split(current_feature_set, y, test_size = 0.50, random_state = 1)

    X_train_fold2 = X_test_fold1
    X_test_fold2 = X_train_fold1
    y_train_fold2 = y_test_fold1
    y_test_fold2 = y_train_fold1

    decision_tree = DecisionTreeClassifier()

    decision_tree.fit(X_train_fold1, y_train_fold1)
    pred_f1 = decision_tree.predict(X_test_fold1)
    decision_tree.fit(X_train_fold2, y_train_fold2)
    pred_f2 = decision_tree.predict(X_test_fold2)

    predictions = np.concatenate([pred_f1, pred_f2])
    y_test = np.concatenate((y_test_fold1, y_test_fold2))

    org_acc = accuracy_score(y_test, predictions)

    # Calculating accuracy for perturbed set
    X_train_fold1, X_test_fold1, y_train_fold1, y_test_fold1 = train_test_split(perturbed_feature_set, y, test_size = 0.50, random_state = 1)

    X_train_fold2 = X_test_fold1
    X_test_fold2 = X_train_fold1
    y_train_fold2 = y_test_fold1
    y_test_fold2 = y_train_fold1

    decision_tree = DecisionTreeClassifier()

    decision_tree.fit(X_train_fold1, y_train_fold1)
    pred_f1 = decision_tree.predict(X_test_fold1)
    decision_tree.fit(X_train_fold2, y_train_fold2)
    pred_f2 = decision_tree.predict(X_test_fold2)

    predictions = np.concatenate([pred_f1, pred_f2])
    y_test = np.concatenate((y_test_fold1, y_test_fold2))

    prime_acc = accuracy_score(y_test, predictions)

    # Compare accuracies
    if prime_acc > org_acc:
        current_feature_set = perturbed_feature_set
        status = "IMPROVED"
        subset = "NEW SUBSET"
        accuracy = prime_acc
        acceptance_probability = "---"
        random_uniform = "---"
    else:
        subset = "ORIGINAL SUBSET"
        accuracy = org_acc
        acceptance_probability = exp((i * -1) * ((org_acc - prime_acc) / org_acc))
        random_uniform = np.random.uniform()
        if random_uniform > acceptance_probability:
            status = "DISCARDED"
        else:
            status = "ACCEPTED"

    print("Iteration: " + str(i))
    print("Subset: " + subset)
    print("Accuracy: " + str(accuracy))
    print("Pr[Accept]: " + str(acceptance_probability))
    print("Random Uniform: " + str(random_uniform))
    print("Status: " + status)
    print("\n")

X_train_fold1, X_test_fold1, y_train_fold1, y_test_fold1 = train_test_split(current_feature_set, y, test_size = 0.50, random_state = 1)

X_train_fold2 = X_test_fold1
X_test_fold2 = X_train_fold1
y_train_fold2 = y_test_fold1
y_test_fold2 = y_train_fold1

decision_tree = DecisionTreeClassifier()

decision_tree.fit(X_train_fold1, y_train_fold1)
pred_f1 = decision_tree.predict(X_test_fold1)
decision_tree.fit(X_train_fold2, y_train_fold2)
pred_f2 = decision_tree.predict(X_test_fold2)

predictions = np.concatenate([pred_f1, pred_f2])
y_test = np.concatenate((y_test_fold1, y_test_fold2))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nAccuracy Score:")
print(accuracy_score(y_test, predictions))
print("\n\n")

