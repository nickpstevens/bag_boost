import numpy as np
from scipy.special import expit
from collections import Counter
from mldata import *


"""
Written by Nick Stevens
10/9/2016
"""


# Useful constants
CLASS_LABEL = -1
EXAMPLE_WEIGHT = 0


class ANN(object):
    LEARNING_RATE = 0.01

    def __init__(self, training_set, validation_set, num_hidden_units, weight_decay_coeff, boosting=False):
        self.num_hidden_units = num_hidden_units
        # Set up the training and validation sets
        self.full_training_set = training_set
        self.full_validation_set = validation_set
        self.training_labels = self.full_training_set[:, [CLASS_LABEL]]
        self.training_examples = self.full_training_set[:, :CLASS_LABEL]
        self.validation_labels = self.full_validation_set[:, [CLASS_LABEL]]
        self.validation_examples = self.full_validation_set[:, :CLASS_LABEL]
        if boosting:
            self.training_example_weights = self.training_examples[:, [EXAMPLE_WEIGHT]]
            self.training_examples = self.training_examples[:, EXAMPLE_WEIGHT+1:]
        else:
            self.training_example_weights = None
        (self.num_training_examples, self.num_features) = self.training_examples.shape
        # Adjust weight-decay coefficient to account for stochastic learning
        self.weight_decay_coeff = weight_decay_coeff / self.num_training_examples
        # Make sure training set and validation set are compatible
        assert self.num_features == self.validation_examples.shape[1]
        # Weight matrices setup
        if self.num_hidden_units != 0:
            self.hidden_weights = np.random.uniform(-0.1, 0.1, (self.num_features, self.num_hidden_units))
            self.output_weights = np.random.uniform(-0.1, 0.1, (self.num_hidden_units, 1))
        else:
            self.hidden_weights = None
            self.output_weights = np.random.uniform(-0.1, 0.1, (self.num_features, 1))
        # Additional matrices
        self.hidden_inputs = None
        self.hidden_outputs = None
        self.output_inputs = None
        self.output_sigmoids = None
        self.output_labels = np.empty(self.training_labels.shape)

    def train(self, num_training_iters, chunk_size=1, convergence_err=float(1e-8), max_iters=10000):
        assert chunk_size > 0
        assert convergence_err >= 0.0
        assert max_iters > 0
        original_weight_decay = self.weight_decay_coeff
        if chunk_size != 1:
            # Rescale the weight-decay coefficient to account for the number of examples
            self.weight_decay_coeff *= chunk_size
        if num_training_iters == 0:
            i = 0
            while not np.array_equal(self.output_labels, self.training_labels):
                output_dl_dw = self.stochastic_learning(chunk_size)
                i += 1
                print('\t' + str(i) + '.\tIteration accuracy:\t' +
                      str(np.sum(self.output_labels == self.training_labels) / float(self.num_training_examples)))
                if (np.absolute(output_dl_dw) < convergence_err).all() or i >= max_iters:
                    # Additional stopping conditions:
                    # Stop iterating if all errors are smaller than the threshold
                    # Also stop if too many iterations have occurred
                    break
        else:
            for i in range(0, num_training_iters):
                self.stochastic_learning(chunk_size)
                print('\t' + str(i+1) + '.\tIteration accuracy:\t' +
                      str(np.sum(self.output_labels == self.training_labels) / float(self.num_training_examples)))
        self.weight_decay_coeff = original_weight_decay  # reset this in case it has been changed

    def stochastic_learning(self, chunk_size):
        """
        For each example in the training set, feed it through the neural network and then use backpropagation to update
        the weights. This is implemented as stochastic learning, but it supports a sort-of "mini-batch" functionality.
        Basically, you can set the number of examples to be passed through the network at a time. Unfortunately, to get
        the best accuracy, this number should be set to 1. I thought this would be a cool feature that speeds things up,
        but it just makes convergence take way longer. So it's pretty much useless.
        """
        output_dl_dw = None
        for i in xrange(0, self.num_training_examples, chunk_size):
            actual_labels = np.array(self.training_labels[i:i+chunk_size], ndmin=2)
            examples = np.array(self.training_examples[i:i+chunk_size, :], ndmin=2)
            self.feedforward(examples, i)
            output_dl_dw = self.backpropagation(actual_labels, examples, i)
        return output_dl_dw

    def feedforward(self, examples, index):
        if self.num_hidden_units != 0:
            # Feed examples through Hidden Layer
            self.hidden_inputs = np.dot(examples, self.hidden_weights)
            self.hidden_outputs = self.sigmoid(self.hidden_inputs)
            # Feed examples through Output Layer
            self.output_inputs = np.dot(self.hidden_outputs, self.output_weights)
        else:
            self.output_inputs = np.dot(examples, self.output_weights)
        self.output_sigmoids = self.sigmoid(self.output_inputs)
        new_labels = self.binary_values(self.output_sigmoids)
        np.put(self.output_labels, index, new_labels)
        return new_labels

    def backpropagation(self, actual_labels, examples, index):
        if self.num_hidden_units != 0:
            output_dl_dw = self.calc_output_dl_dw(actual_labels)
            hidden_dl_dw = self.calc_hidden_dl_dw(examples, output_dl_dw)
            self.update_weights(output_dl_dw, hidden_dl_dw)
        else:
            output_dl_dw = self.calc_output_dl_dw(actual_labels, examples, index)
            self.update_weights(output_dl_dw)
        return output_dl_dw

    def update_weights(self, output_dl_dw, hidden_dl_dw=None):
        self.output_weights -= (self.LEARNING_RATE * (output_dl_dw + self.weight_decay_coeff * self.output_weights))
        if hidden_dl_dw is not None:
            self.hidden_weights -= (self.LEARNING_RATE * (hidden_dl_dw + self.weight_decay_coeff * self.hidden_weights))

    def calc_output_dl_dw(self, actual_labels, inputs=None, index=None):
        """
        Calculates the loss due to the output-layer weights between the output unit and the hidden-layer outputs.
        Returned matrix should have shape (num_hidden_units, 1) or (num_features, 1) if there are no hidden units
        """
        subtracted_term = self.output_sigmoids - actual_labels
        d_sigmoid = self.d_sigmoid(self.output_inputs)
        if inputs is None:
            d_sigmoid_times_inputs = np.dot(self.hidden_outputs.T, d_sigmoid)
        else:
            d_sigmoid_times_inputs = np.dot(inputs.T, d_sigmoid)
        dl_dw = np.dot(d_sigmoid_times_inputs, subtracted_term.T)
        dl_dw_avg = np.sum(dl_dw, axis=1)  # If there are multiple examples, average the partial loss across examples
        dl_dw_avg = np.reshape(dl_dw_avg, (dl_dw_avg.shape[0], -1))
        if self.training_example_weights is not None and index is not None:
            # Incorporate boosting weight into loss
            dl_dw_avg *= (self.training_example_weights[index] * self.num_training_examples)
        return dl_dw_avg

    def calc_hidden_dl_dw(self, examples, output_dl_dw):
        """
        Calculates the loss due to the hidden-layer weights between the hidden layer and the input units.
        This calculation is simplified by the fact that the only downstream unit is the single output unit.
        Returned matrix should have shape (num_features, num_hidden_units)
        """
        d_sigmoid = self.d_sigmoid(self.hidden_inputs)
        d_sigmoid_times_examples = np.dot(examples.T, d_sigmoid)
        quotient = self.output_weights / self.hidden_outputs.T
        downstream = output_dl_dw * quotient
        downstream_sum = np.mean(downstream, axis=1)
        downstream_sum = np.reshape(downstream_sum, (-1, downstream_sum.shape[0]))
        dl_dw = d_sigmoid_times_examples * downstream_sum
        return dl_dw

    def sigmoid(self, x):
        sigmoid = np.copy(x)
        sigmoid[sigmoid < -709] = -709  # Standardize values too large for expit()
        sigmoid[sigmoid > 709] = 709
        sigmoid = expit(sigmoid)  # Efficient sigmoid calculation from scipy
        return sigmoid

    def d_sigmoid(self, x):
        # The derivative of the sigmoid function
        sigmoid = self.sigmoid(x)
        return sigmoid * (1 - sigmoid)

    def binary_values(self, x):
        # Converts matrix values to binary
        bin_x = np.copy(x)
        bin_x[bin_x > 0.5] = 1
        bin_x[bin_x <= 0.5] = 0
        return bin_x

    def evaluate(self):
        # Feeds the validation set through the network
        num_examples = len(self.validation_examples)
        for i in xrange(0, num_examples):
            example = np.array(self.validation_examples[i, :], ndmin=2)
            self.feedforward(example, [i])
        # Return (actual labels, assigned labels)
        return self.validation_labels, self.output_labels[:num_examples]


def main(options):
    assert options is not None
    assert len(options) == 5
    file_base = options[0]
    example_set = parse_c45(file_base)
    schema = example_set.schema

    default_cv_option = 0
    default_num_hidden_units = 20
    default_weight_decay_coeff = 0.01
    default_num_training_iters = 0

    # If 0, use cross-validation. If 1, run algorithm on full sample.
    cv_option = (1 if options[1] == '1' else default_cv_option)
    try:
        num_hidden_units = (int(options[2]) if int(options[2]) >= 0 else default_num_hidden_units)
    except ValueError:
        num_hidden_units = default_num_hidden_units
    try:
        weight_decay_coeff = float(options[3])
    except ValueError:
        weight_decay_coeff = default_weight_decay_coeff
    try:
        num_training_iters = (int(options[4]) if int(options[4]) > 0 else default_num_training_iters)
    except ValueError:
        num_training_iters = default_num_training_iters

    # Create a numpy array from the example set
    example_set = np.array(example_set.to_float(), ndmin=2)
    # Shuffle the set to ensure that it is not ordered by class label
    np.random.seed(12345)
    np.random.shuffle(example_set)
    # Standardize the feature values in the example set
    example_set = standardize(example_set)

    if cv_option == 1:
        accuracy, precision, recall, false_positive_rate \
            = run(example_set, example_set, num_hidden_units, weight_decay_coeff, num_training_iters)
        print('Accuracy:\t' + str("%0.6f" % accuracy))
        print('Precision:\t' + str("%0.6f" % precision))
        print('Recall:\t\t' + str("%0.6f" % recall))
    else:
        fold_set = k_folds_stratified(example_set, 5)
        accuracy_vals, precision_vals, recall_vals, fpr_vals = \
            run_cross_validation(fold_set, num_hidden_units, weight_decay_coeff, num_training_iters)
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


def k_folds_stratified(example_set, k):
    label_dist = Counter(ex[CLASS_LABEL] for ex in example_set)
    label_values = label_dist.keys()
    examples_with_label = [[] for x in xrange(len(label_values))]
    # Get the set of examples for each label
    for example in example_set:
        for label in label_values:
            if example[CLASS_LABEL] == label_values[int(label)]:
                examples_with_label[int(label)].append(example)
                break
    # Group examples by class label
    sorted_examples = []
    for example_subset in examples_with_label:
        sorted_examples += example_subset
    folds = [[] for x in xrange(k)]
    # Distribute sorted examples evenly amongst all k folds
    for i in xrange(0, len(sorted_examples)):
        assigned_fold = i % k
        folds[assigned_fold].append(sorted_examples[i])
    for j in xrange(0, k):
        fold = np.array(folds[j])
        np.random.shuffle(fold)
        folds[j] = fold
    return folds


def run_cross_validation(fold_set, num_hidden_units, weight_decay_coeff, num_training_iters):
    num_folds = len(fold_set)
    assert num_folds != 0
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
        accuracy, precision, recall, false_positive_rate \
            = run(training_set, validation_set, num_hidden_units, weight_decay_coeff, num_training_iters)
        np.put(accuracy_vals, i, accuracy)
        np.put(precision_vals, i, precision)
        np.put(recall_vals, i, recall)
        np.put(fpr_vals, i, false_positive_rate)
    return accuracy_vals, precision_vals, recall_vals, fpr_vals


def run(training_set, validation_set, num_hidden_units, weight_decay_coeff, num_training_iters):
    print('Building ANN')
    ann = ANN(training_set, validation_set, num_hidden_units, weight_decay_coeff)
    print('\nTraining ANN')
    ann.train(num_training_iters)
    print('\nEvaluating ANN performance\n')
    return evaluate_ann_performance(ann)


def evaluate_ann_performance(ann, label_pairs=None):
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    if label_pairs is None:
        actual_labels, assigned_labels = ann.evaluate()
        num_examples = len(actual_labels)
        assert num_examples > 0
        label_pairs = zip(actual_labels, assigned_labels)
    else:
        num_examples = np.shape(label_pairs)[0]
    for labels in label_pairs:
        if labels == (1.0, 1.0):
            true_positives += 1
        elif labels == (1.0, 0.0):
            false_negatives += 1
        elif labels == (0.0, 1.0):
            false_positives += 1
        elif labels == (0.0, 0.0):
            true_negatives += 1
    print('\n\tValidation Set Distribution')
    print('\tTP: ' + str(true_positives))
    print('\tTN: ' + str(true_negatives))
    print('\tFP: ' + str(false_positives))
    print('\tFN: ' + str(false_negatives) + '\n')
    accuracy = float(true_positives + true_negatives) / num_examples
    try:
        precision = float(true_positives) / (true_positives + false_positives)
    except ZeroDivisionError:
        precision = 0.0
    try:
        recall = float(true_positives) / (true_positives + false_negatives)
    except ZeroDivisionError:
        recall = 0.0
    try:
        false_positive_rate = float(false_positives) / (false_positives + true_negatives)
    except ZeroDivisionError:
        false_positive_rate = 0.0
    return accuracy, precision, recall, false_positive_rate


def find_area_under_roc(fpr_vals, tpr_vals):
    assert len(fpr_vals) == len(tpr_vals)
    roc_data = zip(fpr_vals, tpr_vals)
    roc_data = np.sort(roc_data, axis=0)  # Sort by false-positive rate
    first_point = [0, 0]
    last_point = [1, 1]
    roc_data = np.vstack([first_point, roc_data, last_point])
    area = 0
    for i in xrange(1, len(roc_data)):
        height = roc_data[i, 0] - roc_data[i-1, 0]
        if height == 0:
            pass
        base = roc_data[i, 1]
        top = roc_data[i-1, 1]
        area += 0.5 * (base + top) * height
    return area


if __name__ == "__main__":
    main(sys.argv[1:])