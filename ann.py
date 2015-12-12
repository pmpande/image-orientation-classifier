import time
import math
#http://arctrix.com/nas/python/bpnn.py


def summation1(w, inp, no, minimum, maximum):
    result = 0
    for x in range(minimum, maximum):
        result += w[(x, no)]*inp[x]
    return result


def summation2(w, inp, no, minimum, maximum):
    result = 0
    for x in range(minimum, maximum):
        result += w[(no, x)]*inp[x]
    return result


def get_output_vector(value):
    if value == '0':
        return [1.0, 0, 0, 0]
    if value == '90':
        return [0, 1.0, 0, 0]
    if value == '180':
        return [0, 0, 1.0, 0]
    else:
        return [0, 0, 0, 1.0]


def g(val):
    return 1.0/(1.0 + math.exp(-val))


def g_dash(val):
    return g(val)*(1.0 - g(val))


def initialize_weights(w, no_input_nodes, no_hidden_nodes, no_output_nodes):
    for x in range(no_input_nodes):
        for y in range(no_hidden_nodes):
            w[(x, y + no_input_nodes)] = 1.0
    for x in range(no_hidden_nodes):
        for y in range(no_output_nodes):
            w[(x + no_input_nodes, y + no_input_nodes + no_hidden_nodes)] = 1.0
    return w


def update_weights(w, alpha, a, delta, no_input_nodes, no_hidden_nodes, no_output_nodes):
    for x in range(no_input_nodes):
        for y in range(no_hidden_nodes):
            w[(x, y + no_input_nodes)] += a[x] * delta[y + no_input_nodes] * alpha
    for x in range(no_hidden_nodes):
        for y in range(no_output_nodes):
            w[(x + no_input_nodes, y + no_input_nodes + no_hidden_nodes)] += a[x + no_input_nodes] * delta[y + no_input_nodes + no_hidden_nodes] * alpha
    return w


def ann(train_vector):
    no_input_nodes = 192
    no_hidden_nodes = 40
    no_output_nodes = 4
    w = {}
    a = {}
    w = initialize_weights(w, no_input_nodes, no_hidden_nodes, no_output_nodes)
    counter = 0
    for k in range(10):
        cnt=0.0
        for i in train_vector.keys()[1:30000]:
            counter += 1
            output_vector = get_output_vector(train_vector[i][1])
            inp = {}
            delta = {}
            #Forward Propogation
            for x in range(no_input_nodes):
                a[x] = float(train_vector[i][1][x])
            for unit in range(no_input_nodes, no_input_nodes + no_hidden_nodes):
                inp[unit] = summation1(w, a, unit, 0, no_input_nodes)
                a[unit] = g(inp[unit])
            l = []
            for unit in range(no_input_nodes + no_hidden_nodes, no_input_nodes + no_hidden_nodes + no_output_nodes):
                inp[unit] = summation1(w, a, unit, no_input_nodes, no_input_nodes + no_hidden_nodes)
                a[unit] = g(inp[unit])
                l.append(a[unit])
            print l
            max12=l.index(max(l))
            actual=output_vector.index(1)
            if max12==actual:
                cnt+=1
            #Back Propogation
            for unit in range(no_input_nodes + no_hidden_nodes, no_input_nodes + no_hidden_nodes + no_output_nodes):
                delta[unit] = g_dash(inp[unit])*(output_vector[unit - no_input_nodes - no_hidden_nodes] - a[unit])
            for unit in range(no_input_nodes, no_input_nodes + no_hidden_nodes):
                delta[unit] = g_dash(inp[unit])*summation2(w, delta, unit, no_input_nodes + no_hidden_nodes, no_input_nodes + no_hidden_nodes + no_output_nodes)
            # for unit in range(no_input_nodes):
            #     delta[unit] = g_dash(inp[unit])*summation2(w, delta, unit, no_input_nodes, no_input_nodes + no_hidden_nodes)

            #Update Weights
            w = update_weights(w, 0.8, a, delta, no_input_nodes, no_hidden_nodes, no_output_nodes)
            print "Test " + str(counter) + " done"
        print "Accuracy="+str(cnt/300)
    print w


def main():
    start_time = time.time()
    train_vector = {}
    test_vector = {}
    for line in open("train-data.txt"):
        line = line.split()
        train_vector[(line[0], line[1])] = (line[1], line[2:])
    for line in open("test-data.txt"):
        line = line.split()
        test_vector[line[0]] = (line[1], line[2:])
    ann(train_vector)
    print time.time() - start_time


main()