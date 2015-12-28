from collections import Counter


# Function for calculating distance between training and test sample
# Author: Pranav Pande
def euclidean_distance(a, b):
    distance = 0
    for x in xrange(0, len(a)):
        distance += pow((int(a[x]) - int(b[x])), 2)
    return distance


# Function to print confusion matrix
# Author: Pranav Pande
def print_confusion_matrix(c):
    print "Confusion Matrix:"
    inp = ['0  ', '90 ', '180', '270']
    print "   " + " 0 " + " 90 " + " 180 " + " 270 "
    for x in range(4):
        conf_string = inp[x] + " "
        for y in range(4):
            conf_string += str(c[x*90][y*90]) + " "
        print conf_string


# Algorithm to classify given tests using k-nearest neighbour algorithm
# Author: Pranav Pande
def knn(train_vector, test_vector, param):
    correct = 0
    count = 0
    # Matrix for confusion matrix
    c = {0:{0: 0, 90: 0, 180: 0 ,270 : 0}, 90:{0: 0, 90: 0, 180: 0 ,270 : 0}, 180:{0: 0, 90: 0, 180: 0 ,270 : 0}, 270:{0: 0, 90: 0, 180: 0 ,270 : 0}}
    k = int(param)
    # File to write labelled class to each test
    output_file = open("knn_output.txt", "w")
    # Loop over each test sample
    for j in test_vector.keys():
        response = []
        # Calculate euclidean distance of test sample with each training sample
        for i in train_vector.keys():
            response.append([train_vector[i][0], euclidean_distance(train_vector[i][1], test_vector[j][1])])
        # Sort the resultant distance list
        output = sorted(response, key=lambda x: x[1])
        count += 1
        print "Test No: " + str(count) + " Done"

        # Get first k nearest neighbour
        k_list = output[:k]
        end_list = []
        [end_list.append(each[0]) for each in k_list]
        max_count = Counter(end_list)
        # Find the maximum class occuring in first k neighbours
        label = max_count.most_common()
        # Match with expected data
        if label[0][0] == test_vector[j][0]:
            correct += 1
        c[int(test_vector[j][0])][int(label[0][0])] += 1
        output_file.write(str(j) + " " + str(label[0][0]) + "\n")
    output_file.close()
    print_confusion_matrix(c)
    print "Accuracy : " + str(correct*100.0/count)
    