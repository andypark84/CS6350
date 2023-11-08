import pandas as pd
import numpy as np

def replace_values():
    train.loc[train[4]==0] = -1
    test.loc[test[4]==0] = -1

def train_test_split(train,test):
    x_train = train.iloc[:,:-1] 
    y_train = train.iloc[:,-1]
    x_test = test.iloc[:,:-1] 
    y_test = test.iloc[:,-1]
    return x_train, y_train, x_test, y_test

def perceptron_train(X, y, T):
    w = np.zeros(X.shape[1])
    
    for t in range(T):
        misclassified = 0
        for i in range(len(X)):
            if y[i] * np.dot(w, X.iloc[i]) <= 0:
                w += + y[i] * X.iloc[i]
                misclassified += 1
        if misclassified == 0:
            break

    return w

def perceptron_test(X, y, w):
    errors = 0
    for i in range(len(X)):
        if y[i] * np.dot(w, X.iloc[i]) <= 0:
            errors += 1
    return errors / len(y)

def average_perceptron_train(X, y, T):
    w = np.zeros(X.shape[1])
    w_sum = np.zeros(X.shape[1])

    for t in range(T):
        misclassified = 0
        for i in range(len(X)):
            if y[i] * np.dot(w, X.iloc[i]) <= 0:
                w += y[i] * X.iloc[i]
                misclassified += 1
            
            w_sum += w
            
        if misclassified == 0:
            break

    w_avg = w_sum / (T * len(y))

    return w_avg

def average_perceptron_test(X, y, w_avg):
    errors = 0
    for i in range(len(X)):
        if y[i] * np.dot(w_avg, X.iloc[i]) <= 0:
            errors += 1
    return errors / len(y)

if __name__ == "__main__":
    train = pd.read_csv('./data/bank-note/train.csv', header=None)
    test = pd.read_csv('./data/bank-note/test.csv', header=None)
    
    # replace the label 0 to -1 and split train, test data
    replace_values()
    x_train, y_train, x_test, y_test = train_test_split(train, test)

    # standard Perceptron
    learned_weight = perceptron_train(x_train, y_train, 10)
    average_error = perceptron_test(x_test, y_test, learned_weight)

    print("Learned weight vector(standard Perceptron):", learned_weight)
    print("Average prediction error on the test dataset(standard Perceptron):", average_error)

    # average Perceptron
    w_avg = average_perceptron_train(x_train, y_train, 10)
    average_error = average_perceptron_test(x_test, y_test, w_avg)

    print("Learned weight vector(average Perceptron):", w_avg)
    print("Average test error(average Perceptron):", average_error)

