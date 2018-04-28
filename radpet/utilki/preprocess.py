import numpy as np

from sklearn.preprocessing import MinMaxScaler


def scale(data):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled = scaler.fit_transform(data)
    return scaled, scaler


def inverse_scale(y, X, scaler):
    inv = np.concatenate((y, X[:, 1:]), axis=1)  # X is needed in order to rescale back
    inv = scaler.inverse_transform(inv)
    inv = inv[:, 0]

    return inv
