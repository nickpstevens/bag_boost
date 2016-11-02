Bagging and Boosting
Nick Stevens
10/27/2016

This program contains implementations of the "bagging" (bootstrap aggregation) and "boosting" (specifically AdaBoost)
ensemble classification methods.

Currently this only supports the included ANN learning algorithm.

USAGE

The program takes 4 arguments:

    1. The title of the dataset contained in the "data" folder
    2. Cross-validation option. 0 --> use CV; 1 --> run program on the full sample
    3. Name of learning algorithm. "ann" is currently the only supported option.
    4. The number of bagging/boosting iterations

Example: python bag.py "volcanoes" 0 "ann" 10

BAGGING



BOOSTING

Boosting is initiated by training a learner on a set of equally-weighted examples. The weights are then updated so that
misclassified examples have their weights increased, and correctly classified examples have their weights decreased.
For every subsequent iteration, a new learner is created using the original example data and the updated weights.

The ANN has been adapted to handle weighted examples. This was done by multiplying the backpropagated loss of each
example by the example's weight. The resulting term was also multiplied by the number of examples in the training set.
The purpose of this was to scale the loss values so that they aligned more closely with the values from the non-boosted
ANN. The number of examples in the training set is used because the example weights are initially set to the inverse of
this value.