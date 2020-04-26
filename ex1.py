import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

def logistic_func(x):

    ###################################################################
    # YOUR CODE HERE!
    # Output: logistic(x)
    ####################################################################

    return L

def train(X_train, y_train, tol = 10 ** -4):

    LearningRate = 0.05

    ###################################################################
    # YOUR CODE HERE!
    # Output: the weight update result [w_0, w_1, w_2, ...]
    ####################################################################

    return weights

def train_matrix(X_train, y_train, tol = 10 ** -4):

    LearningRate = 0.05

    ###################################################################
    # YOUR CODE HERE!
    # Output: the weight update result [w_0, w_1, w_2, ...]
    ####################################################################

    return weights

def predict(X_test, weights):

    ###################################################################
    # YOUR CODE HERE!
    # The predict labels of all points in test dataset.
    ####################################################################

    return predictions

def plot_prediction(X_test, X_test_prediction):
    X_test1 = X_test[X_test_prediction == 0, :]
    X_test2 = X_test[X_test_prediction == 1, :]
    plt.scatter(X_test1[:, 0], X_test1[:, 1], color='red')
    plt.scatter(X_test2[:, 0], X_test2[:, 1], color='blue')
    plt.show()


#Data Generation
n_samples = 1000

centers = [(-1, -1), (5, 10)]
X, y = make_blobs(n_samples=n_samples, n_features=2, cluster_std=1.8,
                  centers=centers, shuffle=False, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=42)

# Experiments
w = train(X_train, y_train)
X_test_prediction = predict(X_test, w)
plot_prediction(X_test, X_test_prediction)
plot_prediction(X_test, y_test)

wrong = np.count_nonzero(y_test - X_test_prediction)
print ('Number of wrong predictions is: ' + str(wrong))
