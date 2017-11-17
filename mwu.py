import numpy as np
import time

def Oracle(weights, models, X, Y, alpha, findExample):
    return np.array([findExample(weights, models, x, y, alpha) for x, y in zip(X,Y)])

def evaluateCosts(models, V, X, Y):
    return np.array([1 - model.score(X + V, Y) for model in models])

def runMWU(models, T, X, Y, alpha, findExample, epsilon=None):
    num_models = len(models)

    if epsilon is None:
        delta = np.sqrt(4 * np.log(num_models) / float(T))
        epsilon = delta / 2.0
    else:
        delta = 2.0 * epsilon

    print "Running MWU for {} Iterations with Epsilon {}\n".format(T, epsilon)

    print "Guaranteed to be within {} of the minimax value \n".format(delta)

    loss_history = []
    costs = []
    max_acc_history = []
    v = []
    w = []

    w.append(np.ones(num_models) / num_models)

    for t in xrange(T):
        print "Iteration ", t
        print
        start_time = time.time()

        v_t = Oracle(w[t], models, X, Y, alpha, findExample)
        v.append(v_t)

        cost_t = evaluateCosts(models, v_t, X, Y)
        costs.append(cost_t)

        print "Shape of costs matrix", np.array(costs).shape
        avg_acc = np.mean((1 - np.array(costs)), axis=0)
        max_acc = max(avg_acc)
        max_acc_history.append(max_acc)

        loss = np.dot(w[t], cost_t)

        print "Weights, ", w[t], sum(w[t])
        print "Maximum (Average) Accuracy of Classifier ", max_acc
        print "Cost (Before Noise), ", np.array([1 - model.score(X, Y) for model in models])
        print "Cost (After Noise), ", cost_t
        print "Loss, ", loss

        loss_history.append(loss)

        new_w = np.copy(w[t])

        # penalize experts
        for i in xrange(num_models):
            new_w[i] *= (1.0 - epsilon) ** cost_t[i]

        # renormalize weights
        w_sum = new_w.sum()
        for i in xrange(num_models - 1):
            new_w[i] = new_w[i] / w_sum
        new_w[-1] = 1.0 - new_w[:-1].sum()

        w.append(new_w)

        print
        print "time spent ", time.time() - start_time
        print

    return w, v, loss_history, max_acc_history