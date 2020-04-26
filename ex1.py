import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

def logistic_func(x):

    ###################################################################
    # YOUR CODE HERE!
    # Output: logistic(x)
    ####################################################################

    L = 1 / (1 + np.exp(-x))

    return L

def train(X_train, y_train, tol = 10 ** -4):

    LearningRate = 0.05

    ###################################################################
    # YOUR CODE HERE!
    # Output: the weight update result [w_0, w_1, w_2, ...]
    ####################################################################

    dim = len(X_train[0]) + 1
    weights = np.random.randn((dim))

    def g(x):
        #print("g:w = {}".format(weights))
        return np.dot(weights[1:], x.T) + weights[0]

    while True:
        #print("w={}".format(weights))
        w = np.copy(weights)
        w[0] = weights[0] + LearningRate * np.sum((y_train - logistic_func(g(X_train))))
        for i in range(1, dim):
            #print(i)
            #print(weights)
            w[i] = weights[i] + LearningRate * np.sum((y_train - logistic_func(g(X_train))) * X_train.T[i-1])

        diff = weights - w
        diff = np.linalg.norm(diff)
        #print(diff)

        if diff < tol:
            weights = w
            break

        weights = w

    return weights

def train_matrix(X_train, y_train, tol = 10 ** -4):

    LearningRate = 0.05

    ###################################################################
    # YOUR CODE HERE!
    # Output: the weight update result [w_0, w_1, w_2, ...]
    ####################################################################
    X = np.column_stack([np.ones((X_train.shape[0],1)) , X_train])
    weights = np.random.randn((len(X[0])))

    def g(x):
        #print("g:w = {}".format(weights))
        return np.dot(weights[1:], x.T) + weights[0]

    while True:
        w = np.copy(weights)
        w = weights + LearningRate * np.matmul((y_train - logistic_func(np.matmul(X,weights))).T , X)
        if np.linalg.norm(weights - w) < tol:
            weights = w
            break
        weights = w

    return weights

def predict(X_test, weights):

    ###################################################################
    # YOUR CODE HERE!
    # The predict labels of all points in test dataset.
    ####################################################################
    def g(x):
        return np.dot(weights[1:], x.T) + weights[0]


    predictions = logistic_func(g(X_test))
    predictions = np.around(predictions)

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
