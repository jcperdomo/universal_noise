import numpy as np


def shuffleArraysInUnison(a, b, p=None):
    # https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
    assert len(a) == len(b)
    if p is None:
        p = np.random.permutation(len(a))
    return a[p], b[p], p

def generate_data(num_pts, X, Y, models, target_dict=False):
    num_selected = 0
    num_models = len(models)
    resX = []
    resY = []
    for i in xrange(len(X)):
        allCorrect = sum([model.score(X[i:i+1], Y[i:i+1]) for model in models]) == num_models
        if allCorrect:
            if target_dict:
                true_label = np.argmax(Y[i])
                target_labels = target_dict[true_label]
                for l in target_labels:
                    resX.append(X[i])
                    resY.append((np.arange(10) == l).astype(np.float32))
            else:
                resX.append(X[i])
                resY.append(Y[i])
            num_selected += 1
        if num_selected == num_pts:
            break
    if num_selected < num_pts:
        print "Not enough points were correctly predicted by all models"
    return np.array(resX), np.array(resY)


