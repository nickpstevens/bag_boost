Implementations of the bootstrap aggregation and boosting (AdaBoost) ensemble methods.

Currently this only supports the ANN learning algorithm.

Boosting is initiated by training a learner on a set of equally-weighted examples. The weights are then updated so that
misclassified examples have their weights increased, and correctly classified examples have their weights decreased.
For every subsequent iteration, a new learner is created using the original example data and the updated weights.