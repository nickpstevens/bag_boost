from random import seed, randrange
from collections import Counter
import numpy as np

from mldata import *
from ann import ANN, find_area_under_roc, evaluate_ann_performance, k_folds_stratified


"""
Created by Nick Stevens
10/27/2016
"""


# Useful constants
CLASS_LABEL = -1
LEARNING_ALGORITHMS = {'ann': ANN}


def main(options):
    assert options is not None
    assert len(options) == 4
    file_base = options[0]
    example_set = parse_c45(file_base)
    schema = example_set.schema

    default_cv_option = 0
    default_learning_algorithm = ANN
    default_num_bagging_training_iters = 3

    # If 0, use cross-validation. If 1, run algorithm on full sample.
    cv_option = (1 if options[1] == '1' else default_cv_option)
    try:
        learning_algorithm = LEARNING_ALGORITHMS[options[2]] if options[2] in LEARNING_ALGORITHMS.keys() else\
            default_learning_algorithm
    except ValueError:
        learning_algorithm = default_learning_algorithm
    try:
        num_bagging_training_iters = (int(options[3]) if int(options[3]) > 0 else default_num_bagging_training_iters)
    except ValueError:
        num_bagging_training_iters = default_num_bagging_training_iters

    if learning_algorithm is ANN:
        # Learning algorithm is ANN
        num_hidden_units = len(schema) - 1  # The number of features (excluding the class label)
        weight_decay_coeff = 0.01
        num_ann_training_iters = 3  # TODO may need to be adjusted
        if cv_option == 1:
            accuracy, precision, recall, fpr = ann_bag(example_set, example_set, schema, num_hidden_units,
                                                       weight_decay_coeff, num_ann_training_iters,
                                                       num_bagging_training_iters)
            print('Accuracy:\t' + str("%0.6f" % accuracy))
            print('Precision:\t' + str("%0.6f" % precision))
            print('Recall:\t\t' + str("%0.6f" % recall))
        else:
            num_folds = 5
            fold_set = k_folds_stratified(example_set, schema, num_folds)
            accuracy_vals = np.empty(num_folds)
            precision_vals = np.empty(num_folds)
            recall_vals = np.empty(num_folds)
            fpr_vals = np.empty(num_folds)
            for i in xrange(0, num_folds):
                validation_set = fold_set[i]
                training_set = ExampleSet(schema)
                for j in xrange(1, 5):
                    k = (i + j) % 5
                    for example in fold_set[k]:
                        training_set.append(example)
                print('Fold ' + str(i + 1))
                accuracy, precision, recall, fpr = ann_bag(training_set, validation_set, schema, num_hidden_units,
                                                            weight_decay_coeff, num_ann_training_iters,
                                                            num_bagging_training_iters)
                np.put(accuracy_vals, i, accuracy)
                np.put(precision_vals, i, precision)
                np.put(recall_vals, i, recall)
                np.put(fpr_vals, i, fpr)
            accuracy = np.mean(accuracy_vals)
            accuracy_std = np.std(accuracy_vals, ddof=1)
            precision = np.mean(precision_vals)
            precision_std = np.std(precision_vals, ddof=1)
            recall = np.mean(recall_vals)
            recall_std = np.std(recall_vals, ddof=1)
            area_under_roc = find_area_under_roc(fpr_vals, recall_vals)
            print('Accuracy:\t' + str("%0.6f" % accuracy) + '\t' + str("%0.6f" % accuracy_std))
            print('Precision:\t' + str("%0.6f" % precision) + '\t' + str("%0.6f" % precision_std))
            print('Recall:\t\t' + str("%0.6f" % recall) + '\t' + str("%0.6f" % recall_std))
            print('Area under ROC:\t' + str("%0.6f" % area_under_roc) + '\n')
    else:
        raise NotImplementedError


def ann_bag(training_set, validation_set, schema, num_hidden_units,
            weight_decay_coeff, num_ann_training_iters, num_bagging_training_iters):
    iter_labels = np.empty(len(validation_set))
    for i in xrange(0, num_bagging_training_iters):
        print('\nBagging iteration ' + str(i+1))
        replicate_set = bootstrap_replicate(training_set, schema, seed_value=i)
        ann = ANN(replicate_set, validation_set, num_hidden_units, weight_decay_coeff)
        ann.train(num_ann_training_iters)
        iter_labels = np.column_stack((iter_labels, ann.evaluate()[1]))
    voting_labels = np.apply_along_axis(most_common_label, 1, iter_labels)
    assert ann is not None
    actual_labels = ann.validation_labels
    label_pairs = zip(actual_labels, voting_labels)
    accuracy, precision, recall, fpr = evaluate_ann_performance(None, label_pairs)
    return accuracy, precision, recall, fpr


def most_common_label(vector):
    counter = Counter(vector)
    return counter.most_common(1)[0][0]

def bootstrap_replicate(example_set, schema, size=None, seed_value=12345):
    """
    Creates a bootstrap replicate of example_set by sampling with replacement. If creating multiple replicates, input a
    different seed_value for every call to produce different sets.
    """
    num_examples = len(example_set)
    if size is None:
        size = num_examples
    seed(seed_value)
    replicate = ExampleSet(schema)
    for i in xrange(0, size):
        replicate.append(example_set[randrange(num_examples)])
    return replicate


if __name__ == "__main__":
    main(sys.argv[1:])
