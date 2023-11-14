import pandas as pd
import numpy as np
from sklearn.utils import shuffle

def replace_values():
    train.loc[train[4]==0] = -1
    test.loc[test[4]==0] = -1

def train_test_split(train,test):
    x_train = train.iloc[:,:-1] 
    y_train = train.iloc[:,-1]
    x_test = test.iloc[:,:-1] 
    y_test = test.iloc[:,-1]
    return x_train, y_train, x_test, y_test

def svm_primal_sgd(X, y, C, lr_schedule):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0
    max_epoch = 100 

    for epoch in range(max_epoch):
        X, y = shuffle(X, y, random_state = 2023)
        for i in range(n_samples):
            lr = lr_schedule(epoch * n_samples + i + 1)
            xi = X.iloc[i]
            yi = y[i]
            margin = yi * (np.dot(w, xi) + b)
            
            if margin < 1:
                subgradient_w = w / C - yi * xi
                subgradient_b = -yi
            else:
                subgradient_w = w / C
                subgradient_b = 0
            
            w = w - lr * subgradient_w
            b = b - lr * subgradient_b

    return w, b

def lr_schedule_1(t):
    return 0.01 / (1 + 0.01 * 0.1 * t)

def lr_schedule_2(t):
    return 0.01 / (1 + t)

def calculate_accuracy(X, y, w, b):
    predictions = np.sign(np.dot(X, w) + b)
    accuracy = np.mean(predictions == y)
    return accuracy

def schedule_1_error():
    C_dict = [100/873, 500/873, 700/873]

    for C in C_dict:
        
        w, b = svm_primal_sgd(x_train, y_train, C, lr_schedule_1)
        train_accuracy = calculate_accuracy(x_train, y_train, w, b)
        print(f'train accuracy using schedule 1 learning rate & parameter {C}: {train_accuracy}')
        test_accuracy = calculate_accuracy(x_test, y_test, w, b)
        print(f'test accuracy using schedule 1 learning rate & parameter {C}: {test_accuracy}')

def schedule_2_error():
    C_dict = [100/873, 500/873, 700/873]

    for C in C_dict:
       
        w, b = svm_primal_sgd(x_train, y_train, C, lr_schedule_2)
        train_accuracy = calculate_accuracy(x_train, y_train, w, b)
        print(f'train accuracy using schedule 2 learning rate & parameter {C}: {train_accuracy}')
        test_accuracy = calculate_accuracy(x_test, y_test, w, b)
        print(f'test accuracy using schedule 2 learning rate & parameter {C}: {test_accuracy}')

if __name__ == "__main__":
    train = pd.read_csv('./data/bank-note/train.csv', header=None)
    test = pd.read_csv('./data/bank-note/test.csv', header=None)
    
    # replace the label 0 to -1 and split train, test data
    replace_values()
    x_train, y_train, x_test, y_test = train_test_split(train, test)
    
    # Q 2-(a)
    schedule_1_error()
    # Q 2-(b)
    schedule_2_error()