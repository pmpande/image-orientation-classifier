from collections import Counter
import time


def euclideanDistance(A, B):
    distance = 0
    for x in xrange(0,len(A)):
        distance += int(A[x]) - int(B[x])**2
    return distance


def knn():
    trainVector = {}
    testVector = {}
    for line in open("train-data.txt"):
        line = line.split()
        trainVector[(line[0], line[1])] = (line[1], line[2:])
    for line in open("test-data.txt"):
        line = line.split()
        testVector[line[0], line[1]] = (line[1], line[2:])
    correct = 0
    xnt = 0
    for j in testVector.keys()[1:5]:
        response = []
        for i in trainVector.keys():
            response.append([trainVector[i][0], euclideanDistance(trainVector[i][1], testVector[j][1])])
        output = sorted(response, key=lambda x: x[1])
        xnt += 1
        print "Test No: " + str(xnt) + " Done"
        k = 5
        klist = output[:k]
        end_list = []
        [end_list.append(each[0]) for each in klist]
        count = Counter(end_list)
        label = count.most_common()
        if label[0][0] == testVector[j][0]:
            correct += 1
    print correct
    print "Accuracy : " + str(correct*100.0/len(testVector))


start_time = time.time()
knn()
print "Run Time: " + str(time.time() - start_time)