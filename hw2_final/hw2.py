import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def train_test_split(train,test):
    x_train = train.iloc[:,:-1] 
    y_train = train.iloc[:,-1]
    x_test = test.iloc[:,:-1] 
    y_test = test.iloc[:,-1]
    return x_train, y_train, x_test, y_test

def train_bgd(lr):
    # initialize weight to 0
    w = np.zeros(x_train.shape[1])

    # hyperparametes
    learning_rate = lr
    tolerance = 1e-6
    num_iterations = 10000

    # history
    costs = []
    weight_diffs = []

    for i in range(num_iterations):
        predictions =  np.dot(x_train,w)
        
        cost = (1/(2*len(y_train))) * np.sum((predictions - y_train)**2)
        costs.append(cost)

        gradient = (1/len(y_train)) * np.dot(x_train.T, (predictions - y_train))
        w -= learning_rate * gradient
        
        weight_diff = np.linalg.norm(learning_rate * gradient)
        weight_diffs.append(weight_diff)

        if weight_diff < tolerance:
            break

    plt.plot(range(i+1), costs)
    plt.xlabel('Iteration')
    plt.ylabel('Cost Function')
    plt.title(f'Convergence(learning_rate:{lr})')
    plt.show()
    print(f'Converged after {i+1} iterations')
    print("Learned weight vector:", w)

def test_bgd():
    # initialize weight to 0
    w = np.zeros(x_train.shape[1])

    # hyperparametes
    learning_rate = 0.5
    tolerance = 1e-6
    num_iterations = 10000

    # history
    costs = []
    weight_diffs = []

    for i in range(num_iterations):
        predictions =  np.dot(x_train,w)
        
        cost = (1/(2*len(y_train))) * np.sum((predictions - y_train)**2)
        costs.append(cost)

        gradient = (1/len(y_train)) * np.dot(x_train.T, (predictions - y_train))
        w -= learning_rate * gradient
        
        weight_diff = np.linalg.norm(learning_rate * gradient)
        weight_diffs.append(weight_diff)

        if weight_diff < tolerance:
            break

    print(f"Final learned weight vector: {w}, learning rate: 0.5")
    test_pred = np.dot(x_test, w)
    print("Predictions of x_test:", test_pred)

def test_sgd():
    # initialize weight to 0
    w = np.zeros(x_train.shape[1])

    # hyperparameters
    learning_rate = 0.01  
    num_iterations = 10000
    tolerance = 1e-6

    # history
    costs = []

    for i in range(num_iterations):
        indexes = np.arange(len(y_train))
        np.random.shuffle(indexes)

        for index in indexes:
            x = x_train.iloc[index,:]
            y = y_train[index]

            prediction = np.dot(w, x)

            gradient = (prediction - y) * x
            
            w -= learning_rate * gradient

        predictions = np.dot(x_train, w)
        cost = (1 / (2 * len(y_train))) * np.sum((predictions - y) ** 2)
        costs.append(cost)

        if i > 0 and abs(costs[i] - costs[i - 1]) < tolerance:
            print(f"Converged after {i+1} iterations")
            break

    # Plot cost function values over iterations
    plt.plot(range(i + 1), costs)
    plt.xlabel("Iteration")
    plt.ylabel("Cost Function")
    plt.title("Convergence")
    plt.show()

    # The learned weight vector
    print("Learned weight vector:", w)

    test_pred = np.dot(x_test, w)
    test_cost = (1 / (2 * len(y_test))) * np.sum((test_pred - y_test) ** 2)
    print("Cost of test data:", test_cost)

if __name__ == "__main__":
    train = pd.read_csv('./data/concrete-2/concrete/train.csv', header=None)
    test = pd.read_csv('./data/concrete-2/concrete/test.csv', header=None)
    x_train, y_train, x_test, y_test = train_test_split(train, test)

    # Q.4-(a): batch gradient descent algorithm
    learning_rates = [1, 0.5, 0.25, 0.125]
    for lr in learning_rates:
        train_bgd(lr)

    # prediction of test data
    test_bgd()

    # Q.4-(b): stochastic gradient descent algorithm
    test_sgd()