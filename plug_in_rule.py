"""
# plug-in-rule.py
#
# Code to perform the simple Plug-in Rule classifier.
# CS481 Assignment #6
# Martin Miglio
"""

import matplotlib.pyplot as plt
import numpy
from statistics import mean


def read_file(file_name):
    """
    " open data file and import as a float numpy array
    """
    dataset = []

    with open(file_name, 'r', errors='ignore', encoding="utf-8") as open_file:
        lines = open_file.readlines()
        for line in lines:
            values = line.split()
            dataset.append(values)

    training_dataset = numpy.asfarray(dataset, dtype=float)
    return training_dataset


def plot_classes(dataset, title, class_colors=['red', 'blue']):
    """
    # use matplotlib to plot data and color by classes
    """
    for value in dataset:
        plt.scatter(value[0],
                    0,
                    c=class_colors[int(0 if value[1] == 1 else 1)],
                    )
    plt.title(title)
    plt.show()


def classify(dataset):
    """
    # use plug-in method for classification
    """
    predictions = []
    values = dataset[:, 0].tolist()  # split out just values
    value_mean = mean(values)
    weights = 2 * value_mean
    weights_out = -value_mean * value_mean
    for value in values:
        prediction = (weights * value) + weights_out
        prediction = int(prediction / 100)
        predictions.append([value, 2 if prediction >= 2 else 1])
    return numpy.array(predictions)


def score(training_set, classified_set):
    """
    # score the classification of the algorithm versus the training data
    """
    correct_predictions = 0
    for index, training_value in enumerate(training_set):
        if classified_set[index][1] == training_set[index][1]:
            correct_predictions += 1
    accuracy = correct_predictions / len(classified_set)
    return accuracy


if __name__ == "__main__":
    # get the data
    file_name = 'data/homework_classify_train_1D.dat'
    training_dataset = read_file(file_name)

    # plot the pre-classified data
    plot_classes(training_dataset, '1d Pre-Classified Dataset')

    # classify data using plug-in
    classified_dataset = classify(training_dataset)

    # plot newly classified data
    plot_classes(training_dataset, '1d Dataset, Classified by Plug-in')

    # score the plug-in classification
    classification_score = score(training_dataset, classified_dataset)
    print(f'Classification accuracy score: {classification_score}')
