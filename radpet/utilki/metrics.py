import numpy as np

EPS = 1e-10


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true + EPS - y_pred) / y_true + EPS)) * 100
