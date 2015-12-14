from neural_network_classifier import load_trained_data
from neural_network_classifier import test
from neural_network_classifier import nnet


# Neural Network Classifier Algorithm with fine tuned parameters
def best(train_vector, test_vector, param):
    # param contains file name which contains pre-trained data
    # if param is present load trained data and test
    if param is not None:
        weights = load_trained_data(param)
        test(test_vector, 192, 40, 4, weights, "best_output.txt")
    # Else train the network first and then test
    else:
        no_hidden_nodes = 40
        learning_rate = 0.1
        no_of_iterations = 5
        output_file_name = "best_output.txt"
        nnet(train_vector, test_vector, no_hidden_nodes, no_of_iterations, learning_rate, output_file_name)
