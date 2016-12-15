#!/usr/bin/env python
# Author: Christian Diller
# Compares the performance of the fertilized forests boosting implementations
# to the opencv and scikit learn implementations

import os
import sys
sys.path.insert(0, os.path.join('..', '..', 'build', 'bindings', 'python'))

# Classifiers
from cv2 import RTrees, Boost, BOOST_REAL, CV_ROW_SAMPLE, TERM_CRITERIA_MAX_ITER
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from fertilized import Soil, Result_Types

# Evaluation tools
from sklearn.datasets import make_classification
from sklearn.metrics import f1_score
from time import clock

# Output tools
from tabulate import tabulate
from progressbar import ProgressBar, Percentage, Bar, ETA

# Util
import numpy as np
from itertools import product

# Samples creation
def create_samples(n_samples, n_features, n_classes):
    X, Y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_features/2, n_classes=n_classes)
    X = X.astype("float32")
    Y = Y.astype("uint32")
    return X, Y


# Classifiers setup
def setup_opencv_rtrees(n_trees, n_features, max_depth, min_samples_at_leaf):
    parameters = {"max_depth": max_depth,
                  "min_sample_count": min_samples_at_leaf,
                  "nactive_vars": n_features,
                  "max_num_of_trees_in_the_forest": n_trees,
                  "term_crit": (TERM_CRITERIA_MAX_ITER, n_trees, 1)}
    return parameters

def setup_opencv_boost(n_trees, max_depth):
    parameters = {"boost_type": BOOST_REAL,
                  "weak_count": n_trees,
                  "weight_trim_rate": 0,
                  "max_depth": max_depth}
    return parameters

def setup_sklearn_randomForest(n_trees, n_features, max_depth, min_samples_at_leaf):
    forest = RandomForestClassifier(n_estimators=n_trees,
                                    criterion='gini',
                                    max_features=n_features,
                                    max_depth=max_depth,
                                    min_samples_leaf=min_samples_at_leaf,
                                    min_samples_split=2*min_samples_at_leaf,
                                    bootstrap=False)
    return forest

def setup_sklearn_adaBoost(n_trees, n_features, max_depth, min_samples_at_leaf, algorithm):
    base_estimator = DecisionTreeClassifier(criterion="gini",
                                            splitter="best",
                                            max_features=n_features,
                                            max_depth=max_depth,
                                            min_samples_split=2*min_samples_at_leaf,
                                            min_samples_leaf=min_samples_at_leaf)
    boost = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=n_trees, algorithm=algorithm)
    return boost

def setup_fertilized(n_trees, n_features, n_classes, max_depth, min_samples_at_leaf, algorithm, use_boosting_leafman):
    soil = Soil('f', 'f', 'uint', Result_Types.probabilities)
    cls = []  # Deciders
    lm = []   # Leaf managers
    for i in range(n_trees):
        stdFeatureSelect = soil.StandardFeatureSelectionProvider(n_features,   # Selections per node
                                                                 1,    # Data dimensions selected per proposal
                                                                 n_features,   # How many dimensions available (number of features)
                                                                 n_features,   # How many dimensions to use
                                                                 1+i)  # Seed for random number generator
        rcto = soil.ClassificationThresholdOptimizer(True, # fast approximation
                                                     n_classes,
                                                     soil.EntropyGain(soil.InducedEntropy(2.)),
                                                     0.) # gain threshold                                                     
#        rcto = soil.RandomizedClassificationThresholdOptimizer(10,    # Number of thresholds to test per node
#                                                               n_classes,    # Number of classes
#                                                               soil.EntropyGain(soil.ShannonEntropy()),
#                                                               0,    # Minimum gain to continue splitting
#                                                               1,    # Memory step from one annotation to another
#                                                               1+i)  # Seed for random number generator

        # ThresholdDecider
        tClassifier = soil.ThresholdDecider(stdFeatureSelect, soil.AlignedSurfaceCalculator(), rcto)

        if use_boosting_leafman:
            leafMgr = soil.BoostingLeafManager(n_classes)
        else:
            leafMgr = soil.ClassificationLeafManager(n_classes)

        cls.append(tClassifier)
        lm.append(leafMgr)

    training = soil.ClassicTraining(soil.NoBagging())
    if algorithm == "ADABOOST":
        training = soil.BoostedTraining(soil.AdaBoost())
    elif algorithm == "SAMME":
        training = soil.BoostedTraining(soil.Samme())
    elif algorithm == "SAMME.R":
        training = soil.BoostedTraining(soil.Samme_R())

    forest = soil.Forest(max_depth,              # Maximum tree depth
                         min_samples_at_leaf,    # Min samples at leaf
                         2*min_samples_at_leaf,  # Min samples at node
                         n_trees,                # Number of trees
                         cls,                    # Classifiers
                         lm,                     # Leaf managers
                         training)
    return forest


def perform_evaluation(trees, features, samples, n_iterations, n_classes, testing_algorithms):
    if n_classes > 2 and ("opencv_plain" in testing_algorithms or "opencv_adaboost" in testing_algorithms):
        raise Exception("OpenCV does not support more than two classes")
    permutations = list(product(*[trees, features, samples]))
    for (n_trees, n_features, n_samples) in permutations:
        steps = len(testing_algorithms) + 1

        times = [[] for _ in testing_algorithms]
        scores = [[] for _ in testing_algorithms]

        if n_classes > 2:
            f1_average = "macro"
        else:
            f1_average = "binary"

        pbar = ProgressBar(widgets=[Percentage(), Bar(), ETA()], maxval=steps*n_iterations).start()

        print(("Current parameters: n_trees="+str(n_trees)+" n_samples="+str(n_samples)+" n_features="+str(n_features)))

        for index in range(n_iterations):
            current_step = 0
            # Generate samples
            trainX, trainY = create_samples(n_samples, n_features, n_classes)
            testX, testY = create_samples(n_samples, n_features, n_classes)
            pbar.update(index*steps+current_step)
            current_step += 1

            for algorithm in testing_algorithms:
                if algorithm == "fertilized_plain":
                    fertilizedForest = setup_fertilized(n_trees, n_features, n_classes, 1, 2, "PLAIN", False)
                    fertilizedTrainY = np.ascontiguousarray(np.atleast_2d(trainY).T)
                    start = clock()
                    fertilizedForest.fit(trainX, fertilizedTrainY)
                    end = clock()
                    times[current_step-1].append(end - start)
                    prediction = fertilizedForest.predict(testX)
                    scores[current_step-1].append(f1_score(testY, [np.argmax(x) for x in prediction], average=f1_average))
                elif algorithm == "fertilized_samme":
                    fertilizedForest = setup_fertilized(n_trees, n_features, n_classes, 1, 2, "SAMME", False)
                    fertilizedTrainY = np.ascontiguousarray(np.atleast_2d(trainY).T)
                    start = clock()
                    fertilizedForest.fit(trainX, fertilizedTrainY)
                    end = clock()
                    times[current_step-1].append(end - start)
                    prediction = fertilizedForest.predict(testX)
                    scores[current_step-1].append(f1_score(testY, [np.argmax(x) for x in prediction], average=f1_average))
                elif algorithm == "fertilized_samme_r":
                    fertilizedForest = setup_fertilized(n_trees, n_features, n_classes, 1, 2, "SAMME.R", False)
                    fertilizedTrainY = np.ascontiguousarray(np.atleast_2d(trainY).T)
                    start = clock()
                    fertilizedForest.fit(trainX, fertilizedTrainY)
                    end = clock()
                    times[current_step-1].append(end - start)
                    prediction = fertilizedForest.predict(testX)
                    scores[current_step-1].append(f1_score(testY, [np.argmax(x) for x in prediction], average=f1_average))
                elif algorithm == "fertilized_adaboost":
                    fertilizedForest = setup_fertilized(n_trees, n_features, n_classes, 1, 2, "ADABOOST", False)
                    fertilizedTrainY = np.ascontiguousarray(np.atleast_2d(trainY).T)
                    start = clock()
                    fertilizedForest.fit(trainX, fertilizedTrainY)
                    end = clock()
                    times[current_step-1].append(end - start)
                    prediction = fertilizedForest.predict(testX)
                    scores[current_step-1].append(f1_score(testY, [np.argmax(x) for x in prediction], average=f1_average))
                elif algorithm == "fertilized_samme_r_leafman":
                    fertilizedForest = setup_fertilized(n_trees, n_features, n_classes, 1, 2, "SAMME.R", True)
                    fertilizedTrainY = np.ascontiguousarray(np.atleast_2d(trainY).T)
                    start = clock()
                    fertilizedForest.fit(trainX, fertilizedTrainY)
                    end = clock()
                    times[current_step-1].append(end - start)
                    prediction = fertilizedForest.predict(testX)
                    scores[current_step-1].append(f1_score(testY, [np.argmax(x) for x in prediction], average=f1_average))
                elif algorithm == "sklearn_plain":
                    sklearnForest = setup_sklearn_randomForest(n_trees, n_features, 1, 2)
                    start = clock()
                    sklearnForest.fit(trainX, trainY)
                    end = clock()
                    times[current_step-1].append(end - start)
                    prediction = sklearnForest.predict(testX)
                    scores[current_step-1].append(f1_score(testY, prediction, average=f1_average))
                elif algorithm == "sklearn_samme":
                    sklearnBoost = setup_sklearn_adaBoost(n_trees, n_features, 1, 2, "SAMME")
                    start = clock()
                    sklearnBoost.fit(trainX, trainY)
                    end = clock()
                    times[current_step-1].append(end - start)
                    prediction = sklearnBoost.predict(testX)
                    scores[current_step-1].append(f1_score(testY, prediction, average=f1_average))
                elif algorithm == "sklearn_samme_r":
                    sklearnBoost = setup_sklearn_adaBoost(n_trees, n_features, 1, 2, "SAMME.R")
                    start = clock()
                    sklearnBoost.fit(trainX, trainY)
                    end = clock()
                    times[current_step-1].append(end - start)
                    prediction = sklearnBoost.predict(testX)
                    scores[current_step-1].append(f1_score(testY, prediction, average=f1_average))
                elif algorithm == "opencv_plain":
                    p = setup_opencv_rtrees(n_trees, n_features, 1, 2)
                    opencvForest = RTrees()
                    start = clock()
                    opencvForest.train(trainX, CV_ROW_SAMPLE, trainY.astype("int32"), params=p)
                    end = clock()
                    times[current_step-1].append(end - start)
                    prediction = [int(x) for x in [opencvForest.predict(x) for x in testX]]
                    scores[current_step-1].append(f1_score(testY, prediction, average=f1_average))
                elif algorithm == "opencv_adaboost":
                    p = setup_opencv_boost(n_trees, 1)
                    opencvBoost = Boost()
                    start = clock()
                    opencvBoost.train(trainX, CV_ROW_SAMPLE, trainY.astype("int32"), params=p)
                    end = clock()
                    times[current_step-1].append(end - start)
                    prediction = [int(x) for x in [opencvBoost.predict(x) for x in testX]]
                    scores[current_step-1].append(f1_score(testY, prediction, average=f1_average))
                pbar.update(index*steps+current_step)
                current_step += 1

        pbar.finish()

        output = []
        for idx, alg in enumerate(testing_algorithms):
            curr = [alg, str(np.mean(scores[idx])), str(np.mean(times[idx]))]
            output.append(curr)

        table = tabulate(output, headers=["Algorithm", "F1 Score", "Fit Time in s"])
        print(table)


# These are all implementations available for comparison
implementations = ["fertilized_plain",                  # fertilized without boosted training
                   "fertilized_samme", "fertilized_samme_r", "fertilized_adaboost", # fertilized boosting methods
                   "fertilized_samme_r_leafman",        # fertilized with samme_r boosting implementing the original algorithm
                   "sklearn_plain",                     # sklearn without boosted training
                   "sklearn_samme", "sklearn_samme_r",  # sklearn boosting methods
                   "opencv_plain",                      # opencv without boosted training
                   "opencv_adaboost"]                   # opencv adaboost

# Choose number of trees, features, samples, iterations and classes and the implementations to use for comparison
perform_evaluation([25, 50, 75, 100], [50], [5000], 25, 2,
                   ["fertilized_plain", "fertilized_samme", "fertilized_samme_r", "fertilized_adaboost",
                    "fertilized_samme_r_leafman", "opencv_plain", "opencv_adaboost",
                    "sklearn_plain", "sklearn_samme", "sklearn_samme_r"])
