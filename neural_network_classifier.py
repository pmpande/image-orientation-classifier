from math import exp
from random import randrange
from sys import exit


# Summation of all the inputs and corresponding weights for each hidden node
# Author: Pranav Pande
def summation1(w, inp, no, minimum, maximum):
    result = 0
    for x in range(minimum, maximum):
        result += w[(x, no)]*inp[x]
    return result


# Summation of all the outputs and corresponding weights for each hidden node
# Author: Pranav Pande
def summation2(w, inp, no, minimum, maximum):
    result = 0
    for x in range(minimum, maximum):
        result += w[(no, x)]*inp[x]
    return result


# Get the target output for a given actual class value
# Author: Pranav Pande
def get_output_vector(value):
    if value == '0':
        return [1.0, 0, 0, 0]
    elif value == '90':
        return [0, 1.0, 0, 0]
    elif value == '180':
        return [0, 0, 1.0, 0]
    elif value == '270':
        return [0, 0, 0, 1.0]
    else:
        "Error in fetching input"
        exit()


# Sigmoid function
# Author: Pranav Pande
def g(val):
    return 1.0/(1.0 + exp(-val))


# Derivative of sigmoid function
# Author: Pranav Pande
def g_dash(val):
    return g(val)*(1.0 - g(val))


# Initialize all weights with a random value between (-1, 1)
# Author: Pranav Pande
def initialize_weights(w, no_input_nodes, no_hidden_nodes, no_output_nodes):
    for x in range(no_input_nodes):
        for y in range(no_hidden_nodes):
            w[(x, y + no_input_nodes)] = randrange(-1, 1)*(0.1)
    for x in range(no_hidden_nodes):
        for y in range(no_output_nodes):
            w[(x + no_input_nodes, y + no_input_nodes + no_hidden_nodes)] = randrange(-1, 1)*(0.1)
    return w


# Update weights based on the learning rate, error, inputs and weights
# Author: Pranav Pande
def update_weights(w, alpha, a, delta, no_input_nodes, no_hidden_nodes, no_output_nodes):
    for x in range(no_input_nodes):
        for y in range(no_hidden_nodes):
            w[(x, y + no_input_nodes)] += a[x] * delta[y + no_input_nodes] * alpha
    for x in range(no_hidden_nodes):
        for y in range(no_output_nodes):
            w[(x + no_input_nodes, y + no_input_nodes + no_hidden_nodes)] += a[x + no_input_nodes] * delta[y + no_input_nodes + no_hidden_nodes] * alpha

    return w


# Store all the weights attained after training in a text file
# Author: Pranav Pande
def save_trained_data(w, file_name):
    weights = open(file_name, "w")
    for k, v in w.items():
        weights.write(str(k[0])+" "+str(k[1])+" "+str(v)+"\n")
    weights.close()


# Load all the weights attained after training from the file
# Author: Pranav Pande
def load_trained_data(file_name):
    weights = open(file_name, "r")
    w = {}

    for line in weights:
        line = line.split()
        w[(int(line[0]), int(line[1]))] = float(line[2])
    return w


# Function to print the confusion matrix
# Author: Pranav Pande
def print_confusion_matrix(c):
    print "Confusion Matrix:"
    inp = ['0  ', '90 ', '180', '270']
    print "   " + " 0 " + " 90 " + " 180 " + " 270 "
    for x in range(4):
        conf_string = inp[x] + " "
        for y in range(4):
            conf_string += str(c[str(x)][str(y)]) + " "
        print conf_string


# Function to train the neural network
# Author: Pranav Pande and Nishant Shah
def train(train_vector, no_input_nodes, no_hidden_nodes, no_output_nodes, no_of_iterations, learning_rate):

    w = {}
    w = initialize_weights(w, no_input_nodes, no_hidden_nodes, no_output_nodes)
    # Loop over all the training samples multiple times
    for k in range(no_of_iterations):
        cnt = 0.0
        counter = 0
        # Loop over each training sample
        for i in train_vector.keys():
            counter += 1

            # Get target vector
            target_vector = get_output_vector(i[1])

            # Activation dictionary
            a = {}
            # Input dictionary
            inp = {}
            # Error dictionary
            delta = {}

            # Forward propagation
            for x in range(no_input_nodes):
                a[x] = float(train_vector[i][1][x])/255
            for unit in range(no_input_nodes, no_input_nodes + no_hidden_nodes):
                inp[unit] = summation1(w, a, unit, 0, no_input_nodes) + 1
                a[unit] = g(inp[unit])

            output_vector = []
            for unit in range(no_input_nodes + no_hidden_nodes, no_input_nodes + no_hidden_nodes + no_output_nodes):
                inp[unit] = summation1(w, a, unit, no_input_nodes, no_input_nodes + no_hidden_nodes) + 1
                a[unit] = g(inp[unit])
                output_vector.append(a[unit])

            max_index = output_vector.index(max(output_vector))
            actual = target_vector.index(1.0)
            # Check if output found is same as the target
            if max_index == actual:
                cnt += 1

            # Back Propagation
            for unit in range(no_input_nodes + no_hidden_nodes, no_input_nodes + no_hidden_nodes + no_output_nodes):
                delta[unit] = g_dash(inp[unit])*(target_vector[unit - no_input_nodes - no_hidden_nodes] - a[unit])

            for unit in range(no_input_nodes, no_input_nodes + no_hidden_nodes):
                delta[unit] = g_dash(inp[unit])*summation2(w, delta, unit, no_input_nodes + no_hidden_nodes, no_input_nodes + no_hidden_nodes + no_output_nodes)

            # Update Weights
            w = update_weights(w, learning_rate, a, delta, no_input_nodes, no_hidden_nodes, no_output_nodes)
            print "Train sample " + str(counter) + " done | Accuracy: " + "%.4f" % (cnt/counter*100.0) + " %"
        print cnt
    return w

# Author: Pranav Pande and Nishant Shah
def test(test_vector, no_input_nodes, no_hidden_nodes, no_output_nodes, w, output_file_name):
    cnt = 0.0
    a = {}
    # Matrix for confusion matrix
    c = {'0':{'0': 0, '1': 0, '2': 0 ,'3' : 0},'1':{'0': 0, '1': 0, '2': 0 ,'3' : 0},'2':{'0': 0, '1': 0, '2': 0 ,'3' : 0},'3':{'0': 0, '1': 0, '2': 0 ,'3' : 0}}
    counter = 0
    # File to store the resultant test sample with its label
    output_file = open(output_file_name, "w")
    # File to store incorrect examples for analysis
    incorrect = open("incorrect.txt", "w")
    # Loop over each test sample
    for i in test_vector.keys():
        inp = {}
        target_vector = get_output_vector(test_vector[i][0])
        # Find output using the trained network
        for x in range(no_input_nodes):
            a[x] = float(test_vector[i][1][x])/255
        for unit in range(no_input_nodes, no_input_nodes + no_hidden_nodes):
            inp[unit] = summation1(w, a, unit, 0, no_input_nodes) + 1
            a[unit] = g(inp[unit])
        output_vector = []
        for unit in range(no_input_nodes + no_hidden_nodes, no_input_nodes + no_hidden_nodes + no_output_nodes):
            inp[unit] = summation1(w, a, unit, no_input_nodes, no_input_nodes + no_hidden_nodes) + 1
            a[unit] = g(inp[unit])
            output_vector.append(a[unit])

        max_index = output_vector.index(max(output_vector))
        actual = target_vector.index(1.0)
        counter += 1
        output_file.write(str(i) + " " + str(max_index*90) + "\n")
        c[str(actual)][str(max_index)] += 1
        # Check if the output found is same as target
        if max_index == actual:
            cnt += 1
        else:
            if actual == 0:
                actual = '0'
            elif actual == 1:
                actual = '90'
            elif actual == 2:
                actual = '180'
            elif actual == 3:
                actual = '270'
            else:
                pass
            if max_index == 0:
                max_index = '0'
            elif max_index == 1:
                max_index = '90'
            elif max_index == 2:
                max_index = '180'
            elif max_index == 3:
                max_index = '270'
            else:
                pass
            incorrect.write(i+" "+actual+" "+max_index+"\n")
        print "Test " + str(counter) + " done | Accuracy: " + "%.4f" % (cnt/counter*100.0) + " %"
    incorrect.write("Confusion "+str(c)+"\n")
    incorrect.close()
    output_file.close()
    print_confusion_matrix(c)
    print "Accuracy: " + "%.4f" % (cnt/counter*100.0) + " %"

# Author: Pranav Pande
def nnet(train_vector, test_vector, no_hidden_nodes, no_of_iterations, learning_rate, output_file_name):

    no_input_nodes = 192
    no_output_nodes = 4

    print "Training Data..."
    w = train(train_vector, no_input_nodes, no_hidden_nodes, no_output_nodes, no_of_iterations, learning_rate)

    save_trained_data(w, "trained_weights.txt")
    weights = load_trained_data("trained_weights.txt")
    print "Testing Data..."
    test(test_vector, no_input_nodes, no_hidden_nodes, no_output_nodes, weights, output_file_name)
