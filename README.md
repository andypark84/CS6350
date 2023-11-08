# CS6350
This is a machine learning library developed by Andrew Park for  CS5350/6350 in University of Utah.
# Linear Regression
For the Linear Regression, when you run the run.sh file, it will run the functions train_bgd(lr), test_bgd(), and test_sgd() sequentially.
The function train_bgd(lr) will plot the graphs of convergence and return the learned weight vectors for each of the learning rates(1, 0.5, 0.25, 0.125).
The function test_bgd() will make a prediction of the test data using the learning rate(0.5) and return the learned weight vetor.
The function test_sgd() will plot the graph of the cost function and return learned weight vector and cost of the test data. 
# Perceptron
For the Perceptron, when you run the run.sh file, it will run the functions replace_values(), train_test_split(train, test), percepton_train(x_train, y_train, 10), perceptron_test(x_test, y_test, learned_weight), average_perceptron_train(x_train, y_train, 10), average_perceptron_test(x_test, y_test, w_avg) sequentially.
The function replace_values() replaces the values of label 0 to -1 in both the train and test data.
The function percepton_train(x_train, y_train, 10) will set the maximum number of epochs to 10 and train the train data using the standard perceptron algorithm. It returns the learned weight vector.
The function percepton_test(x_test, y_test, learned_weight) will use the learned weight vector to return the average prediction error on the test data.
The function average_percepton_train(x_train, y_train, 10) will set the maximum number of epochs to 10 and train the train data using the average perceptron algorithm. It returns the average learned weight vector.
The function average_percepton_test(x_test, y_test, w_avg) will use the average learned weight vector to return the average prediction error on the test data.
