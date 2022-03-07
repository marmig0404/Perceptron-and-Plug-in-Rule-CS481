"""
# perceptron.py
#
# Code to train the Perceptron.
# CS481 Assignment #6
# Martin Miglio
#
# README
#   This program will display 3 graphs:
#       One of the initial training data
#       One of pre-classified testing data
#       One of the testing data classified by the trained perceptron
#   To use this program, install dependencies, run file, close each plot to step the program
# 
# NOTE
#   This program, using random inital values, will sometimes land on a classifier which doesn't
#   perform well. Run the program again if the classifier appears not to work.
#   Because the classifier is linear and the training data is not, the classifier cannot achieve
#   perfect classification
"""

import random
import matplotlib.pyplot as plt
import numpy


def read_file(file_name):
    """
    # read_file function takes a file name and
    #   returns a numpy array of the data
    """
    training_dataset = []
    # set x0 to ones for weight 1 to be the bias
    training_dataset.append(['1', '1', '1'])
    with open(file_name, 'r', encoding="utf-8") as open_file:
        lines = open_file.readlines()
        for (index, line) in enumerate(lines):
            split = line.split()
            training_dataset.append(split)
    return numpy.asfarray(training_dataset, dtype=float)


def activation(value, weight):
    """
    # activation checks a value against the function weights to classify
    """
    activation_function = value[0]*weight[0] + value[1]*weight[1]
    return 1 if activation_function >= 0 else -1


def train(dataset, input_weights):
    """
    # train takes in a dataset and inital weights to train those weights using
    #   the perceptron algorithm for classification
    """
    learning_rate = 0.1
    learned = False
    iteration = 0
    while not learned:
        global_error = 0.0
        for value in dataset:
            activation_on_value = activation(value, input_weights)
            if value[2] != activation_on_value:
                step_error = value[2] - activation_on_value
                input_weights[0] += learning_rate*step_error*value[0]
                input_weights[1] += learning_rate*step_error*value[1]
                global_error += abs(step_error)
            iteration += 1
            if global_error == 0:
                learned = True
    print(f'Performed {iteration} training steps')
    return input_weights


def test_weights(dataset, trained_weights):
    """
    # test values in dataset against weights
    """
    guesses = []
    for value in dataset:
        guess = activation(value, trained_weights)
        guesses.append([value[0], value[1], guess])
    return numpy.array(guesses)


def plot_classes(dataset, title, class_colors=['red', 'blue']):
    """
    # use matplotlib to plot data and color by classes
    """
    for value in dataset:
        plt.scatter(value[0],
                    value[1],
                    c=class_colors[int(0 if value[2] == 1 else 1)],
                    )
    plt.title(title)
    plt.show()


def normalize_data(dataset, max=None, min=None):
    """
    # normalize data to between -1 and 1
    """
    if max is None or min is None:
        max = numpy.amax(dataset)
        min = numpy.amin(dataset)

    normalized_dataset = [[1, 1, 1]]
    for value in dataset:
        normalized_dataset.append([
            ((2*(value[0] - max))-1) / (max - min),
            ((2*(value[1] - max))-1) / (max - min),
            -1 if value[2] == 1 else 1,
        ])
    return numpy.array(normalized_dataset)


def denormalize_data(dataset, max, min):
    """
    # inverse of the normalize function
    """
    denormalized_dataset = []
    for value in dataset:
        denormalized_dataset.append([
            ((value[0] * (max-min)) + (2 * max) + 1)/2,
            ((value[1] * (max-min)) + (2 * max) + 1)/2,
            1 if value[2] == -1 else 2,
        ])
    return numpy.delete(numpy.array(denormalized_dataset), 0, 0)


if __name__ == '__main__':

    # read training data
    training_dataset_file_name = 'data/homework_classify_train_2D.dat'
    training_dataset = read_file(training_dataset_file_name)

    # plot preview of training data
    class_values = numpy.unique(training_dataset[:, 2])
    plot_classes(training_dataset, "Pre-Classified Training Data")

    # normalize training data between -1 & 1
    normalized_training_dataset = normalize_data(training_dataset)

    # generate random weights to begin
    input_weights = [random.uniform(0, 1), random.uniform(0, 1)]
    print(f'Initial weights: {input_weights}')

    # find trained weights against normalized training data
    trained_weights = train(normalized_training_dataset, input_weights)
    print(f'Trained weights: {trained_weights}')

    # read testing data
    testing_dataset_file_name = 'data/homework_classify_test_2D.dat'
    testing_dataset = read_file(testing_dataset_file_name)

    # normalize testing data between -1 & 1
    normalized_testing_dataset = normalize_data(
        testing_dataset,
        numpy.amax(training_dataset),
        numpy.amin(training_dataset)
    )

    # test the trained weights against the test data
    normalized_test_results = test_weights(
        normalized_testing_dataset, trained_weights)

    # denormalize the test results to put them in the same domain as
    #   the test data
    test_results = denormalize_data(
        normalized_test_results,
        numpy.amax(training_dataset),
        numpy.amin(training_dataset)
    )

    # show plots comparing test data classification to trained
    #   perceptron's classification
    plot_classes(testing_dataset, "Pre-Classifed Test Data")
    plot_classes(test_results, "Test Data Classified by Trained Perceptron")
