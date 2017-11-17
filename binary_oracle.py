from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
from sklearn.svm import LinearSVC
from sklearn import preprocessing
from cvxopt import matrix, solvers
from itertools import product

def computeDistsToPlanes(models, X, Y):
    dists = []
    min_max = []
    num_models = len(models)
    num_points = len(X)
    norms = [np.linalg.norm(model.coef_) for model in models]
    for i in xrange(num_points):
        d = []
        for j in xrange(num_models):
            # the reshaping happens so sklearn doesn't complain
            isCorrect = models[j].predict(X[i].reshape(1,-1))[0] == Y[i]
            '''
            I check if things are already correctly predicted when subsetting, not sure what the 
            behavior should be for the case of tryRegion
            '''
            if isCorrect:
                d.append(abs((np.dot(models[j].coef_, X[i]) + models[j].intercept_)) / norms[j])
            else:
                d.append(np.nan)
        min_max.append((min(d), max(d)))
        dists.append(d)
    return np.array(dists).reshape(num_points, num_models), np.array(min_max).reshape(num_points, 2)

def tryRegionBinary(models, signs, x, delta=1e-10):
    """
    finds a vector in the region denoted by the signs vector
    """
    P = matrix(np.identity(x.shape[0]))
    q = matrix(np.zeros(x.shape[0]))
    h = []
    G = []
    num_models = len(models)
    for i in xrange(num_models):
        coef, intercept = models[i].coef_, models[i].intercept_
        ineq_val  = -1.0 * delta + signs[i] * (np.dot(coef, x) + intercept)
        h.append(ineq_val[0])
        G.append(-1.0 * signs[i] * coef.reshape(-1,))
    h = matrix(h)
    G = matrix(np.array(G))
    solvers.options['show_progress'] = False
    sol = solvers.qp(P, q, G, h)
    if sol['status'] == 'optimal':
        v = np.array(sol['x']).reshape(-1,)
        perturbed_x = np.array(x + v).reshape(1, -1)
        is_desired_sign = [models[i].predict(perturbed_x)[0] == signs[i] for i in xrange(num_models)]
        if sum(is_desired_sign) == num_models:
            return v
        else:
            return tryRegionBinary(models, signs, x, delta * 1.5)
    else:
        return None


def findExampleBinary(weights, models, x, y, alpha):
    candidates = []

    # we should only take into consideration models that we could feasibly trickm
    dists, _ = computeDistsToPlanes(models, x.reshape(1, -1), y.reshape(1, -1))
    feasible_models = [models[i] for i in xrange(len(models)) if dists[0][i] < alpha]

    num_models = len(feasible_models)
    # can't trick anything
    if num_models == 0:
        return np.zeros(x.shape)

    rel_weights = np.array([weights[i] for i in xrange(len(models)) if dists[0][i] < alpha])

    signs_values = []
    for signs in product([-1.0, 1.0], repeat=num_models):  # iterate over all possible regions
        is_misclassified = np.equal(-1.0 * y * np.ones(num_models), signs)  # y = -1, or 1
        value = np.dot(is_misclassified, rel_weights)
        signs_values.append((signs, value))

    values = sorted(set([value for signs, value in signs_values]), reverse=True)
    for value in values:
        feasible_candidates = []
        for signs in [sign for sign, val in signs_values if val == value]:
            v = tryRegionBinary(feasible_models, signs, x)
            if v is not None:
                norm = np.linalg.norm(v)
                if norm <= alpha:
                    feasible_candidates.append((v, norm))
        # amongst those with the max value, return the one with the minimum norm
        if feasible_candidates:
            # break out of the loop since we have already found the optimal answer
            return min(feasible_candidates, key=lambda x: x[1])[0]