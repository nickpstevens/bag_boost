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
    default_num_boosting_training_iters = 3

    # If 0, use cross-validation. If 1, run algorithm on full sample.
    cv_option = (1 if options[1] == '1' else default_cv_option)
    try:
        learning_algorithm = LEARNING_ALGORITHMS[options[2]] if options[2] in LEARNING_ALGORITHMS.keys() else\
            default_learning_algorithm
    except ValueError:
        learning_algorithm = default_learning_algorithm
    try:
        num_boosting_training_iters = (int(options[3]) if int(options[3]) > 0 else default_num_boosting_training_iters)
    except ValueError:
        num_boosting_training_iters = default_num_boosting_training_iters

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
        num_ann_training_iters = 500  # TODO may need to be adjusted
        if cv_option == 1:
            accuracy, precision, recall, fpr = ann_boost(example_set, example_set, num_hidden_units,
                                                         weight_decay_coeff, num_ann_training_iters,
                                                         num_boosting_training_iters)
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
                print('Fold ' + str(i + 1))
                accuracy, precision, recall, fpr = ann_boost(training_set, validation_set, num_hidden_units,
                                                             weight_decay_coeff, num_ann_training_iters,
                                                             num_boosting_training_iters)
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
            print('Area under ROC:\t' + str("%0.6f" % aroc) + '\n')
    else:
        raise NotImplementedError


def standardize(example_set):
    """
    Replaces each feature value x in the example set with (x - mean(x)) / standard_deviation(x)
    Any NaN values caused by 0s in the standard deviation are just replaced with 0.
    Uses ddof=1 because this is calculating the sample standard deviation.
    """
    labels = example_set[:, [CLASS_LABEL]]
    feature_values = example_set[:, :CLASS_LABEL]
    standardized = (feature_values - np.mean(feature_values, axis=0)) / np.std(feature_values, axis=0, ddof=1)
    standardized = np.nan_to_num(standardized)
    return np.column_stack((standardized, labels))


def ann_boost(training_set, validation_set, num_hidden_units,
              weight_decay_coeff, num_ann_training_iters, num_boosting_training_iters):
    # Add a column to the front of the example matrix containing the initial weight for each example
    example_weights = np.full((training_set.shape[0], 1), 1.0 / len(training_set))
    anns = []
    alphas = []
    for i in xrange(0, num_boosting_training_iters):
        print('\nBoosting Iteration ' + str(i + 1))
        weighted_training_set = np.column_stack((example_weights, training_set))
        ann = ANN(weighted_training_set, validation_set, num_hidden_units, weight_decay_coeff, boosting=True)
        ann.train(num_ann_training_iters)
        actual_labels = ann.training_labels
        assigned_labels = ann.output_labels
        error = weighted_training_error(example_weights, actual_labels, assigned_labels)
        alpha = classifier_weight(error)
        anns.append(ann)
        alphas.append(alpha)
        example_weights = update_example_weights(example_weights, alpha, actual_labels, assigned_labels)
    alphas = np.array(alphas)
    print(alphas.T)
    vote_labels = weighted_vote_labels(anns, alphas)
    assert ann is not None
    actual_labels = ann.validation_labels
    label_pairs = zip(actual_labels, vote_labels)
    accuracy, precision, recall, fpr = evaluate_ann_performance(None, label_pairs)
    return accuracy, precision, recall, fpr


def weighted_training_error(example_weights, actual_labels, assigned_labels):
    error = 0.0
    for i in xrange(0, len(example_weights)):
        if actual_labels[i] != assigned_labels[i]:
            error += example_weights[i]
    return error


def classifier_weight(error):
    if error == 0.0:
        return float('inf')
    elif error >= 0.5:
        return 0
    else:
        return 0.5 * np.log((1-error) / float(error))


def update_example_weights(example_weights, alpha, actual_labels, assigned_labels):
    # Replace 0 with -1 in labels
    actual_copy = np.copy(actual_labels)
    actual_copy[actual_copy == 0.0] = -1.0
    assigned_copy = np.copy(assigned_labels)
    assigned_copy[assigned_copy == 0.0] = -1.0
    label_signs = actual_copy * assigned_copy
    updated_weights = example_weights * np.exp(-alpha * label_signs)
    weight_sum = np.sum(updated_weights)
    updated_weights /= weight_sum
    return updated_weights


def weighted_vote_labels(anns, alphas):
    all_labels = np.empty((anns[0].validation_labels.shape[0], len(anns)))
    for i in xrange(0, len(anns)):
        iter_labels = anns[i].evaluate()[1].flatten()
        all_labels[:, i] = iter_labels
    vote_labels = np.zeros((all_labels.shape[0], 1))
    alpha_sum = np.sum(alphas)
    for i in xrange(0, len(alphas)):
        alpha = float(alphas[i])
        vote_labels += np.array((alpha / alpha_sum) * all_labels[:, i], ndmin=2).T
    # Map weighted vote results to 1 and 0
    vote_labels[vote_labels > 0.5] = 1
    vote_labels[vote_labels <= 0.5] = 0
    return vote_labels


if __name__ == "__main__":
    main(sys.argv[1:])
