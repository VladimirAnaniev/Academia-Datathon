import numpy as np

EPS = 1e-10


def mape(y_true, y_pred):
    return np.mean(np.abs((y_true + EPS - y_pred) / y_true + EPS)) * 100

def ds(y_true, y_pred):
    dts = 0

    for t in range(1, len(y_true)):
        if (y_true[t] - y_true[t-1]) * (y_pred[t] - y_pred[t-1]) > 0:
            dts += 1
    return 100 / len(y_true) * dts

def m(avg_absolute_errors):
    return np.sum(avg_absolute_errors) / len(avg_absolute_errors)

def r(avg_absolute_errors):
    m = m(avg_absolute_errors)
    
    def s():
        squares = 0
       
        for i in range(len(avg_absolute_errors)):
            squares += np.square(avg_absolute_errors[i] - m)
        
        return np.sqrt(squares / (len(avg_absolute_errors) - 1))
    
    
    return s() / m
    