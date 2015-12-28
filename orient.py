# Names: 1. Pranav Pande (pmpande)
#        2. Nishant Shah (nishshah)
# We have implemented KNN and ANN in seperate files and calling their appropriate methods in this file.
# Repository also includes trained weights and incorrect outputs from ANN using best configuration.
# 1) KNN
# In KNN we are taking Euclidian distance.
# But for sake of less complexity in running time, we are not finding the square root for finding distance, and keeping
# square value of it.
# Training KNN takes a lot of time in local machine(Around 80-90 mins) as we haven't done any optimization.
# 2) ANN
# We started ANN with constant weight 1 to all weights but this resulted in convergence of fitness too early and also
# for every output node.
# So we explored resources to know what should be ideal initial weight distribution and what number of hidden layers
# should be.
# We then came up with a solution of initial weight to be random number between (-1/sqrt(i),1/sqrt(i)) where i is the
# number of input nodes
# and number of hidden nodes to be mean of input and output nodes. But this solution didn't work out well and required
# long time for training.
# So we went for trials and erros for initial weight distribution and number of hidden nodes.
# All results of these experiments are well documented in PDF report.
# Every training of ANN also produces a file trained_weights which eases multiple running of test for same configured
# neural network.
# ->trained_weights which contains weights of every edge.
# Every test run produces a file "incorrect" which contains images classified incorrectly.
# About calculation:
# We are not using matrix multiplication or any external library to speed up our code.
# Our single training completes running in 2-3 mins.
# 3) best
# For best case, we are using ANN. Configuration and result of it is explained in PDF report.
# We have cached the weights of this ANN in model_file. So running the best program will test the test data instantly.
import sys
import time
from k_nearest_neighbour import knn
from neural_network_classifier import nnet
from best_method import best
start_time = time.time()

# Author: Pranav Pande
def main(train_file, test_file, algorithm, param):
    # Dictionary to store data from training and test files
    train_vector = {}
    test_vector = {}

    # Opening training file
    train = open(train_file, "r")
    # Opening test file
    test = open(test_file, "r")

    # Read training file
    for line in train:
        line = line.split()
        train_vector[(line[0], line[1])] = (line[1], line[2:])
    # Read test file
    for line in test:
        line = line.split()
        test_vector[line[0]] = (line[1], line[2:])

    # Call algorithm to classify test inputs
    if algorithm == 'knn':
        knn(train_vector, test_vector, param)
    elif algorithm == 'nnet':
        no_hidden_nodes = int(param)
        learning_rate = 0.5
        no_of_iterations = 3
        output_file_name = "nnet_output.txt"
        nnet(train_vector, test_vector, no_hidden_nodes, no_of_iterations, learning_rate, output_file_name)
    elif algorithm == 'best':
        file_name = param
        best(train_vector, test_vector, file_name)
    else:
        print "Unknown Algorithm"

param1 = sys.argv[1]
param2 = sys.argv[2]
param3 = sys.argv[3]
try:
    param4 = sys.argv[4]
except:
    param4 = None
main(param1, param2, param3, param4)
print "Execution time in minutes: " + str((time.time() - start_time)/60.0)