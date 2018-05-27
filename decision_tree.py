import numpy as np
import math
import random
import pickle
'''import pydot
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import os'''
import sys


# Decision tree class
class Tree:

    def __init__(self):
        self.op = None
        self.kids = np.array([])
        self.node_class = None


'''class Tree:
    # total number of trees
    tree_count = 0

    # initilize class members
    def __init__(self):
        self.op = None
        self.kids = np.array([])
        self.node_class = None

        # increment tree count
        __class__.tree_count += 1
        # node_id is a unque identifier need for tree visualisation
        self.node_id = __class__.tree_count

    # visualise the tree and save graph to png as filename
    # does nothing is pydot is not imported (needs graphvis installed)
    def visualise(self, filename):
        if 'pydot' in sys.modules:
            # pydot graph for tree visualisation
            __class__.vis_graph = pydot.Dot(graph_type='digraph')
            self.vis_node()
            self.vis_edge()
            __class__.vis_graph.write_png(filename)
            print("Saved to current directory: " + filename)

    # add all of the nodes to vis_graph
    def vis_node(self):
        if self.kids.size != 0:
            # add node with tree.op as label
            __class__.vis_graph.add_node(pydot.Node(str(self.node_id),
                                                    label=str(self.op)))
            # recurse through kids
            self.kids[0].vis_node()
            self.kids[1].vis_node()
        # no kids - add as leaf
        else:
            __class__.vis_graph.add_node(pydot.Node(str(self.node_id),
                                                    label=str(self.node_class), style="filled", fillcolor="green"))

    # add all of the edges to vis_graph
    def vis_edge(self):
        if self.kids.size != 0:
            # negative decision
            __class__.vis_graph.add_edge(pydot.Edge(self.node_id,
                                                    self.kids[0].node_id, label='N', color='red'))
            # positive decision
            __class__.vis_graph.add_edge(pydot.Edge(self.node_id,
                                                    self.kids[1].node_id, label='Y', color='blue'))
            # recurse through kids
            self.kids[0].vis_edge()
            self.kids[1].vis_edge()'''


# remapping y vector according to the label
def remapping(binary_t, label):
    result = list(map(lambda element: 1 if element == label else 0, binary_t))
    return result


# find the mode of array
# if there are more than two modes, the final mode will be chosen randomly
def majority_value(targets):
    #target_list = targets.flatten().tolist()
    #target_list = targets.tolist()
    one = np.count_nonzero(targets)
    zero = len(targets) - one
    #one = targets.count(1)
    #zero = targets.count(0)
    if one > zero:
        return 1
    else:
        return 0


# check if all the elements in an array are the same
def same_value(binary_t):
    first = binary_t[0]
    for value in binary_t:
        if value != first:
            return False
    return True


def calculate_entropy(p, n):
    # consider the situation p == 0, which means there are no positive cases
    # in such situation, we need to assign it to 0, otherwise 0 cannot be divided
    if p == 0:
        first_term = 0
    else:
        first_term = -(p / (p + n)) * math.log2(p / (p + n))
    if n == 0:
        second_term = 0
    else:
        second_term = -(n / (p + n)) * math.log2(n / (p + n))
    return first_term + second_term


# calculate the remainder
def calculate_rmd(p0, n0, p1, n1):
    first_term = (p0 + n0) / (p0 + p1 + n0 + n1) * calculate_entropy(p0, n0)
    second_term = (p1 + n1) / (p0 + p1 + n0 + n1) * calculate_entropy(p1, n1)
    return first_term + second_term


# calculate the information gain
def calculate_ig(examples, attr, binary_target):
    p0 = 0
    p1 = 0
    n0 = 0
    n1 = 0
    for i in range(0, len(binary_target)):
        if binary_target[i] == 1:
            if examples[i][attr] == 0:
                p0 += 1
            else:
                p1 += 1
        else:
            if examples[i][attr] == 0:
                n0 += 1
            else:
                n1 += 1
    return calculate_entropy(p0 + p1, n0 + n1) - calculate_rmd(p0, n0, p1, n1)


# calculate information gain for each attribute and choose the best one
def choose_best_decision_attribute(examples, attributes, binary_target):
    best_attributes = 0
    best_ig = -1
    for i in attributes:
        current_ig = calculate_ig(examples, i, binary_target)
        if current_ig > best_ig:
            best_ig = current_ig
            best_attributes = i
    return best_attributes


# delete examples other than those corresponding to the specified attribute value
def find_new_example(example, attr, value):
    '''print(example)
    print(attr)
    print(value)'''
    counter = 0
    for a in range(0, example.shape[0]):
        '''print(a)
        print(attr)'''
        if example[a][attr] == value:
            counter += 1
    new_example = np.zeros((counter, example.shape[1]), dtype='int64')
    counter = 0
    for b in range(0, example.shape[0]):
        if example[b][attr] == value:
            new_example[counter] = example[b]
            counter += 1
    return new_example


# delete target whose corresponding attribute value is not the same as the specified one
def find_new_target(example, targets, attr, value):
    counter = 0
    for a in range(0, example.shape[0]):
        if example[a][attr] == value:
            counter += 1
    new_target = np.zeros(counter, dtype='int64')
    counter = 0
    for b in range(0, example.shape[0]):
        if example[b][attr] == value:
            new_target[counter] = targets[b]
            counter += 1
    return new_target


# update attribute by deleting the best attribute
def delete_best_attribute(attr, best_attribute):
    counter = 0
    while counter < attr.size and attr[counter] != best_attribute:
        counter += 1
    return np.delete(attr, counter, 0)


# check if each row in the matrix 'examples' is identical
def same_examples(array):
    if array.shape[0] <= 1:
        return False
    if array.shape[0] > 1:
        for a in range(1, array.shape[0]):
            if not np.array_equal(array[a], array[a - 1]):
                return False
    return True


def calculate_target_entropy(target):
    target_array = np.array(target)
    target_list = target_array.flatten().tolist()
    num_zeros = target_list.count(0)
    num_ones = target_list.count(1)
    return calculate_entropy(num_ones, num_zeros)


def decision_tree_learning(example, attributes, binary_target):
    node = Tree()
    # all examples have the same value of binary_target
    if same_value(binary_target):
        node.node_class = binary_target[0]
        return node

    # if the attribute array is empty, or the entropy is close to 0
    elif attributes.size == 0 or calculate_target_entropy(binary_target) < ENTROPY_LEVEL:
        node.node_class = majority_value(binary_target)
        return node

    else:
        # find the best attribute
        best_attribute = choose_best_decision_attribute(example, attributes, binary_target)
        # initiate two sub trees
        node.op = best_attribute
        node.kids = np.array([Tree(), Tree()])  # could be improved by not hard-code with two sub-trees only

        # iterate through all possible values of the best-attribute
        for i in [0, 1]:  # could be improved by not hard-code with only two options
            # update examples and target
            new_example = find_new_example(example, best_attribute, i)
            new_target = find_new_target(example, binary_target, best_attribute, i)

            # create a leaf or sub-tree›
            if new_example.size == 0 or same_examples(new_example):
                node.kids[i].node_class = majority_value(binary_target)
            else:
                node.kids[i] = decision_tree_learning(new_example,
                                                      delete_best_attribute(attributes, best_attribute),
                                                      new_target)
    return node


def find_leaf(tree, example):
    if tree.node_class is not None:
        return tree.node_class
    else:
        if example[tree.op] == 1:
            return find_leaf(tree.kids[1], example)
        elif example[tree.op] == 0:
            return find_leaf(tree.kids[0], example)


# This function classify the room according to a vector which contains the results for each tree
# If there is more than one label for an example, a label is randomly chosen
def room_classify(targets):
    if targets.count(1) == 1:
        for i in range(len(targets)):
            if targets[i] == 1:
                return i + 1
    elif targets.count(0) == ROOM_SIZE:
        return random.choice(np.arange(1, ROOM_SIZE + 1))
    else:
        result = []
        for i in range(len(targets)):
            if targets[i] == 1:
                result.append(i + 1)
        return random.choice(result)


# This function returns an array with the size of 6, which has 6 trees according to 5 room type labels
def six_trees_creation(examples, binary_targets):
    attributes = np.arange(0, x[0].size)
    rooms = np.arange(1, ROOM_SIZE + 1)
    decision_tree = []
    for index in range(len(rooms)):
        new_target = remapping(binary_targets, rooms[index])
        tree = decision_tree_learning(examples, attributes, new_target)
        decision_tree.append(tree)
    return decision_tree


def confusion_matrix(test_y, predictions):
    for i in range(ROOM_SIZE):  # five room types
        conf_m = np.zeros(shape=(ROOM_SIZE, ROOM_SIZE))
        for i in range(len(test_y)):
            row = test_y[i] - 1
            col = predictions[i] - 1
            conf_m[row][col] += 1
    return conf_m


def recall_rate(conf_matrix):
    rate = []
    for i in range(ROOM_SIZE):
        TP = conf_matrix[i][i]
        FN = sum(conf_matrix[i]) - TP
        rate.append(TP / (TP + FN))
    return rate


def precision_rate(conf_matrix):
    rate = []
    for i in range(ROOM_SIZE):
        TP = conf_matrix[i][i]
        rate.append(TP / sum(conf_matrix[:, i]))
    return rate


def f_measure(r, p, alpha):
    return (1 + alpha) * (r * p) / (r + alpha * p)


def classification_rate(conf_matrix):
    numerator = 0
    total = sum(sum(conf_matrix))
    for i in range(ROOM_SIZE):
        numerator += conf_matrix[i][i]
    return numerator / total


def error_rate(predictions, correct_values):
    incorrect = 0
    size = len(predictions)
    for i in range(size):
        if predictions[i] != correct_values[i]:
            incorrect += 1
    return incorrect / size


def random_forest_creation(examples, binary_t, no_trees):
    data = np.column_stack((examples, binary_t))
    forest = []
    for i in range(no_trees):
        samples = np.random.randint(0, len(data), int(len(data) / 2))
        samples = np.unique(samples)
        samples_data = data[samples.tolist()]
        six_trees = six_trees_creation(samples_data[:, :-1], samples_data[:, -1])
        forest.append(six_trees)
    return forest


def test_forest(random_forest, test_x):
    predictions = []
    for example in test_x:
        result = np.zeros(ROOM_SIZE)
        for six_tree in random_forest:
            for i in range(len(six_tree)):
                if find_leaf(six_tree[i], example) == 1:
                    result[i] += 1
        predictions.append(result.argmax() + 1)
    return predictions


def tree_evaluation(examples, binary_targets, no_trees):
    forests = []
    examples_size = len(examples)
    fold_size = int(examples_size / NO_OF_FOLD)
    final_matrix = np.zeros(shape=(ROOM_SIZE, ROOM_SIZE))
    recall_r = np.zeros(ROOM_SIZE)
    precision_r = np.zeros(ROOM_SIZE)
    accuracy = 0
    errors = np.zeros(NO_OF_FOLD)
    for no_fold in range(NO_OF_FOLD):
        test_x = examples[(no_fold * fold_size):((no_fold + 1) * fold_size), :]
        test_y = binary_targets[(no_fold * fold_size):((no_fold + 1) * fold_size)]

        training_x = np.append(examples[:(no_fold * fold_size)], examples[((no_fold + 1) * fold_size):], axis=0)
        training_y = np.append(binary_targets[:(no_fold * fold_size)], binary_targets[((no_fold + 1) * fold_size):],
                               axis=0)

        random_forest = random_forest_creation(training_x, training_y, no_trees)
        forests.append(random_forest)
        predictions = test_forest(random_forest, test_x)

        c_matrix = confusion_matrix(test_y, predictions)
        final_matrix += c_matrix
        recall_r += recall_rate(c_matrix)
        precision_r += precision_rate(c_matrix)
        accuracy += classification_rate(c_matrix)
        errors[no_fold] = error_rate(predictions, test_y)

    final_matrix = final_matrix / 10
    recall_r /= NO_OF_FOLD
    precision_r /= NO_OF_FOLD
    f1 = f_measure(recall_r, precision_r, 1)
    accuracy /= NO_OF_FOLD
    print('\nConfusion matrix:')
    print(final_matrix)
    print('\nRecall rate:')
    print(recall_r)
    print('\nPrecision rate:')
    print(precision_r)
    print('\nF1 Measure:')
    print(f1)
    print('\nError rates:')
    print(errors)
    print('\nAccuracy:')
    print(accuracy * 100, '%')

    return accuracy


def plus_one(y):
    result = y
    for i in range(len(result)):
        result[i] += 1
    return result


# loading pickle file
def load_pickle(filename):
    print("Loading pickle file: " + filename)
    with open(filename, 'rb') as f:
        data = pickle.load(f)
        return data


'''def plot_graph():
    t = np.arange(10, 100, 10)
    result = []
    for value in t:
        accu = tree_evaluation(x, y, value)
        result.append(accu)

    plt.plot(t, result)
    plt.show()

    filename = 'tree1.png'
    six_trees[0].visualise(filename)
    filename = 'tree2.png'
    six_trees[1].visualise(filename)
    filename = 'tree3.png'
    six_trees[2].visualise(filename)
    filename = 'tree4.png'
    six_trees[3].visualise(filename)
    filename = 'tree5.png'
    six_trees[4].visualise(filename)'''


ROOM_SIZE = 5
NO_OF_FOLD = 10
ENTROPY_LEVEL = 0.01
NO_OF_ESTIMATORS = 20
x = np.load("object_result.dat")
y = np.load("types_result.dat")
# print(y)
y = plus_one(y)

# shuffle the data
indices = np.random.permutation(x.shape[0])
x = x[indices]
y = y[indices]

# display the whole matrix
np.set_printoptions(threshold=np.nan)
# print(x)
# print(y)

# tree_evaluation(x, y, 70)
# forest = random_forest_creation(x, y, 70)
# six_trees = six_trees_creation(x, y)
# Save random forest
