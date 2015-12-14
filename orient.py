import sys
import time
from k_nearest_neighbour import knn
from neural_network_classifier import nnet
from best_method import best


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


start_time = time.time()
param1 = sys.argv[1]
param2 = sys.argv[2]
param3 = sys.argv[3]
try:
    param4 = sys.argv[4]
except:
    param4 = None
main(param1, param2, param3, param4)
print "Execution time in minutes: " + str((time.time() - start_time)/60.0)