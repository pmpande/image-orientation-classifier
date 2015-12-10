import math
#
# Black = [0,0,0]
# White = [255,255,255]
# Red = [255,0,0]
# Lime = [0,255,0]
# Blue = [0,0,255]
# Yellow = [255,255,0]
# Aqua = [0,255,255]
# Magenta = [255,0,255]
# Silver = [192,192,192]
# Gray = [128,128,128]
# Maroon = [128,0,0]
# Olive = [128,128,0]
# Green = [0,128,0]
# Purple = [128,0,128]
# Teal = [0,128,128]
# Navy = [0,0,128]
#
# color = Black
# print "Black - " + str((color[0]*65536)+(color[1]*256)+color[2])
#
# color = White
# print "White - " + str((color[0]*65536)+(color[1]*256)+color[2])
#
# color = Red
# print "Red - " + str((color[0]*65536)+(color[1]*256)+color[2])
#
# color = Lime
# print "Lime - " + str((color[0]*65536)+(color[1]*256)+color[2])
#
# color = Blue
# print "Blue - " + str((color[0]*65536)+(color[1]*256)+color[2])
#
# color = Yellow
# print "Yellow - " + str((color[0]*65536)+(color[1]*256)+color[2])
#
# color = Aqua
# print "Aqua - " + str((color[0]*65536)+(color[1]*256)+color[2])
#
# color = Magenta
# print "Magenta - " + str((color[0]*65536)+(color[1]*256)+color[2])
#
# color = Silver
# print "Silver - " + str((color[0]*65536)+(color[1]*256)+color[2])
#
# color = Gray
# print "Gray - " + str((color[0]*65536)+(color[1]*256)+color[2])
#
# color = Maroon
# print "Maroon - " + str((color[0]*65536)+(color[1]*256)+color[2])
#
# color = Olive
# print "Olive - " + str((color[0]*65536)+(color[1]*256)+color[2])
#
# color = Green
# print "Green - " + str((color[0]*65536)+(color[1]*256)+color[2])
#
# color = Purple
# print "Purple - " + str((color[0]*65536)+(color[1]*256)+color[2])
#
# color = Teal
# print "Teal - " + str((color[0]*65536)+(color[1]*256)+color[2])
#
# color = Navy
# print "Navy - " + str((color[0]*65536)+(color[1]*256)+color[2])


def euclideanDistance(A, B):
    distance = 0
    for x in range(len(A)):
        distance += pow((A[x] - B[x]), 2)
    return math.sqrt(distance)


trainVector = []
testVector = []
all_lines = [line.split() for line in open("train-data.txt")]
for line in all_lines:
    trainVector.append(line)
all_lines = [line.split() for line in open("test-data.txt")]
for line in all_lines:
    testVector.append(line)
lengthTrain = len(trainVector[0])
lengthTest = len(testVector[0])
dictTrain = {}
dictTest = {}
for i in range(0, len(trainVector)):
    for j in range(1, lengthTrain):
        trainVector[i][j] = int(trainVector[i][j])
    dictTrain[trainVector[i][0]] = trainVector[i][1]
for i in range(0, len(testVector)):
    for j in range(1, lengthTest):
        testVector[i][j] = int(testVector[i][j])
    dictTest[testVector[i][0]] = testVector[i][1]
result = []
for j in range(0, 2): #len(testVector)
    response = []
    for i in range(0, len(trainVector)):
        A = [trainVector[i][item] for item in range(2, lengthTrain)]
        B = [testVector[j][item] for item in range(2, lengthTest)]
        response.append((trainVector[i][0], euclideanDistance(A, B)))
    output = sorted(response, key=lambda x:x[1])
    print "Test " + str(j) + " Done"
result.append(output)
k = 3
correct = 0
for j in range(0, len(testVector)):
    zero = 0
    quarter = 0
    half = 0
    three_fourth = 0
    for i in range(0, k):
        if dictTrain[result[j][i][0]] == 0:
            zero += 1
        elif dictTrain[result[j][i][0]] == 90:
            quarter += 1
        elif dictTrain[result[j][i][0]] == 180:
            half += 1
        elif dictTrain[result[j][i][0]] == 270:
            three_fourth += 1
    label = None
    max = 0
    if zero > max:
        label = 0
        max = zero
    if quarter > max:
        label = 90
        max = quarter
    if half > max:
        label = 180
        max = half
    if three_fourth > max:
        label = 270
        max = three_fourth

    if label == dictTest[testVector[j][0]]:
        correct += 1
print "Accuracy : " + str(correct*100/len(testVector))
