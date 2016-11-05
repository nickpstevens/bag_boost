from collections import Counter
import numpy as np

from mldata import *
from ann import ANN, standardize, find_area_under_roc, evaluate_ann_performance, k_folds_stratified, flip_labels_with_probability


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

    # Create a numpy array from the example set
    example_set = np.array(example_set.to_float(), ndmin=2)
    # Shuffle the set to ensure that it is not ordered by class label
    np.random.seed(12345)
    np.random.shuffle(example_set)
    # Standardize the feature values in the example set
    example_set = standardize(example_set)

    if learning_algorithm is ANN:
        num_hidden_units = 0  # Perceptron
        weight_decay_coeff = 0.01
        num_ann_training_iters = 0
        p = 0.0  # Only increase to introduce noise
        if cv_option == 1:
            accuracy, precision, recall, fpr = ann_bag(example_set, example_set, num_hidden_units,
                                                       weight_decay_coeff, num_ann_training_iters,
                                                       num_bagging_training_iters)
            print('Accuracy:\t' + str("%0.6f" % accuracy))
            print('Precision:\t' + str("%0.6f" % precision))
            print('Recall:\t\t' + str("%0.6f" % recall))
        else:
            num_folds = 5
            fold_set = k_folds_stratified(example_set, num_folds)
            accuracy_vals = np.empty(num_folds)
            precision_vals = np.empty(num_folds)
            recall_vals = np.empty(num_folds)
            fpr_vals = np.empty(num_folds)
            for i in xrange(0, num_folds):
                validation_set = np.array(fold_set[i])
                training_set = []
                for j in xrange(1, 5):
                    k = (i + j) % 5
                    for example in fold_set[k]:
                        training_set.append(example)
                training_set = np.array(training_set)
                training_set = flip_labels_with_probability(training_set, p)
                print('Fold ' + str(i + 1))
                accuracy, precision, recall, fpr = ann_bag(training_set, validation_set, num_hidden_units,
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
            aroc = find_area_under_roc(fpr_vals, recall_vals)
            print('Accuracy:\t' + str("%0.6f" % accuracy) + '\t' + str("%0.6f" % accuracy_std))
            print('Precision:\t' + str("%0.6f" % precision) + '\t' + str("%0.6f" % precision_std))
            print('Recall:\t\t' + str("%0.6f" % recall) + '\t' + str("%0.6f" % recall_std))
            print('Area Under ROC:\t' + str("%0.6f" % aroc) + '\n')
            print('Fold Errors:\t' + str([float(str("%0.6f" % (1-x))) for x in accuracy_vals]) + '\n')
    else:
        raise NotImplementedError


def ann_bag(training_set, validation_set, num_hidden_units,
            weight_decay_coeff, num_ann_training_iters, num_bagging_training_iters):
    iter_labels = None
    example_weights = np.full((training_set.shape[0], 1), 1.0 / len(training_set))
    for i in xrange(0, num_bagging_training_iters):
        print('\nBagging Iteration ' + str(i+1))
        replicate_set = bootstrap_replicate(training_set, seed_value=i)
        weighted_replicate_set = np.column_stack((example_weights, replicate_set))
        ann = ANN(weighted_replicate_set, validation_set, num_hidden_units, weight_decay_coeff, weighted_examples=True)
        ann.train(num_ann_training_iters, convergence_err=0.5)
        if iter_labels is not None:
            iter_labels = np.column_stack((iter_labels, ann.evaluate()[1]))
        else:
            iter_labels = ann.evaluate()[1]
    voting_labels = np.apply_along_axis(most_common_label, 1, iter_labels)
    assert ann is not None
    actual_labels = ann.validation_labels
    label_pairs = zip(actual_labels, voting_labels)
    accuracy, precision, recall, fpr = evaluate_ann_performance(None, label_pairs)
    return accuracy, precision, recall, fpr


def most_common_label(vector):
    counter = Counter(vector)
    return counter.most_common(1)[0][0]


def bootstrap_replicate(example_set, size=None, seed_value=12345):
    """
    Creates a bootstrap replicate of example_set by sampling with replacement. If creating multiple replicates, input a
    different seed_value for every call to produce different sets.
    """
    num_examples = len(example_set)
    if size is None:
        size = num_examples
    np.random.seed(seed_value)
    replicate = np.empty([size, np.shape(example_set)[1]])
    for i in xrange(0, size):
        random_ex = example_set[np.random.randint(0, num_examples), :]
        replicate[i, :] = random_ex
    return replicate


if __name__ == "__main__":
    main(sys.argv[1:])
