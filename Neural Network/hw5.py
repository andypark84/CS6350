import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def train_test_split(train,test):
    x_train = train.iloc[:,:-1] 
    y_train = train.iloc[:,-1]
    x_test = test.iloc[:,:-1] 
    y_test = test.iloc[:,-1]
    return x_train, y_train, x_test, y_test

x = np.array([[1,1]])
y = np.array([1])
W1 = np.array([[-2,2],[-3,3]])
W2 = np.array([[-2,2],[-3,3]])
W3 = np.array([[2],[-1.5]])
b1 = np.array([[-1,1]])
b2 = np.array([[-1,1]])
b3 = np.array([[-1]])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def compute_gradient(X, y):
    z2 = np.dot(X, W1) + b1
    a2 = sigmoid(z2)

    z3 = np.dot(a2, W2) + b2
    a3 = sigmoid(z3)

    z4 = np.dot(a3, W3) + b3
    y_hat = sigmoid(z4)

    error = y - y_hat
        
    delta4 = error  

    z3_error = delta4.dot(W3.T)
    delta3 = z3_error * sigmoid_derivative(a3)

    z2_error = delta3.dot(W2.T)
    delta2 = z2_error * sigmoid_derivative(a2)

    W3_gradient = np.dot(a3.T, delta4)
    W2_gradient = np.dot(a2.T, delta3)
    W1_gradient = np.dot(X.T, delta2)

    print(f'W1 gradient:{W1_gradient}')
    print(f'W2 gradient:{W2_gradient}')
    print(f'W3 gradient:{W3_gradient}') 

class CustomNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))

        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.a2 = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x* (1 - x)

    def forward(self, X):
        self.z1 = np.dot(X.values, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, y_hat, learning_rate):
        # Output layer
        delta2 = y_hat - y
        dW2 = np.dot(self.a1.T, delta2)
        db2 = np.sum(delta2, axis=0, keepdims=True)

        # Hidden layer
        delta1 = np.dot(delta2, self.W2.T) * self.sigmoid_derivative(self.a1)
        dW1 = np.dot(X.T.values.reshape(-1,1), delta1)
        db1 = np.sum(delta1, axis=0, keepdims=True)

        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    def train(self, X, y, epochs, lr_schedule):
        loss_history = []
        train_error_history = []
        num_updates = 0

        for epoch in range(epochs):
            shuffle_idx = np.random.permutation(len(X))
            X_shuffled = X.iloc[shuffle_idx]
            y_shuffled = y[shuffle_idx]

            for i in range(len(X_shuffled)):
                y_hat = self.forward(X_shuffled.iloc[i,:])

                self.backward(X_shuffled.iloc[i,:], y_shuffled[i], y_hat, lr_schedule(num_updates))
                num_updates += 1

                loss = np.mean((y_shuffled[i] - y_hat) ** 2)
                loss_history.append(loss)
            
        return loss_history

class CustomNeuralNetwork_2:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.zeros((input_size, hidden_size))
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.zeros((hidden_size, output_size))
        self.b2 = np.zeros((1, output_size))
        
        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.a2 = None

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.z1 = np.dot(X.values, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2

    def backward(self, X, y, y_hat, learning_rate):
        # Output layer
        delta2 = y_hat - y
        dW2 = np.dot(self.a1.T, delta2)
        db2 = np.sum(delta2, axis=0, keepdims=True)

        # Hidden layer
        delta1 = np.dot(delta2, self.W2.T) * self.sigmoid_derivative(self.a1)
        dW1 = np.dot(X.T.values.reshape(-1,1), delta1)
        db1 = np.sum(delta1, axis=0, keepdims=True)

        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    def train(self, X, y, epochs, lr_schedule):
        loss_history = []
        train_error_history = []
        num_updates = 0

        for epoch in range(epochs):
            shuffle_idx = np.random.permutation(len(X))
            X_shuffled = X.iloc[shuffle_idx]
            y_shuffled = y[shuffle_idx]

            for i in range(len(X_shuffled)):
                y_hat = self.forward(X_shuffled.iloc[i,:])

                self.backward(X_shuffled.iloc[i,:], y_shuffled[i], y_hat, lr_schedule(num_updates))
                num_updates += 1

                loss = np.mean((y_shuffled[i] - y_hat) ** 2)
                loss_history.append(loss)
            
        return loss_history

def lr_schedule(initial_lr, d, t):
    return initial_lr / (1 + initial_lr * d * t)

# hyperparameters
widths = [5, 10, 25, 50, 100]
gamma = 0.1
d = 0.01
epochs = 10


if __name__ == "__main__":
    train = pd.read_csv('./data/bank-note/train.csv', header=None)
    test = pd.read_csv('./data/bank-note/test.csv', header=None)
    x_train, y_train, x_test, y_test = train_test_split(train, test)
    
    compute_gradient(x,y)
    
    for width in widths:
        model = CustomNeuralNetwork(input_size=x_train.shape[1], hidden_size=width, output_size=1)

        loss_history = model.train(x_train, y_train, epochs, lambda t: lr_schedule(gamma, d, t))

        plt.plot(loss_history, label=f'Width: {width}')

        y_pred_train = model.forward(x_train).ravel()
        train_error = np.mean((y_train - y_pred_train) ** 2)
        print(f"Train Error using Width {width}: {train_error}")
        
        y_pred_test = model.forward(x_test).ravel()
        test_error = np.mean((y_test - y_pred_test) ** 2)
        print(f"Test Error using Width {width}: {test_error}")

    plt.xlabel('Number of Updates')
    plt.ylabel('Loss')
    plt.title('Loss History using different widths(weights: Standard Gaussian distribution)')
    plt.legend()
    plt.show()

    for width in widths:
        model = CustomNeuralNetwork_2(input_size=x_train.shape[1], hidden_size=width, output_size=1)

        loss_history = model.train(x_train, y_train, epochs, lambda t: lr_schedule(gamma, d, t))

        plt.plot(loss_history, label=f'Width: {width}')

        y_pred_train = model.forward(x_train).ravel()
        train_error = np.mean((y_train - y_pred_train) ** 2)
        print(f"Train Error using Width {width}: {train_error}")
        
        y_pred_test = model.forward(x_test).ravel()
        test_error = np.mean((y_test - y_pred_test) ** 2)
        print(f"Test Error using Width {width}: {test_error}")

    plt.xlabel('Number of Updates')
    plt.ylabel('Loss')
    plt.title('Loss History using different widths(weights: 0)')
    plt.legend()
    plt.show()