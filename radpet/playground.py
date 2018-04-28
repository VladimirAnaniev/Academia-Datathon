import pandas as pd
from keras.models import load_model

from radpet.baseline.coin_model_trainer import get_features, preprocess
from radpet.utilki.preprocess import inverse_scale


def get_data():
    return pd.DataFrame({
        'price': [9920.23, 1],
        'marketCap': [167795601372, 2],
        'CirculatingSupply': [16914487, 3]
    }, index=[0, 1])


def run():
    MODEL_PATH = './baseline/only_price/1442/checkpoints/weights.02-0.01.hdf5'

    model = load_model(MODEL_PATH)

    coin_data = get_data()

    features = get_features(coin_data).values
    features, scaler = preprocess(features)
    features = features.values

    X = features[:, :-1]
    X = X.reshape((X.shape[0], 1, X.shape[1]))

    print(X)

    yhat = model.predict(X)

    X = X.reshape((X.shape[0], X.shape[2]))

    inv_yhat = inverse_scale(yhat, X, scaler)

    print(yhat, inv_yhat)


if __name__ == '__main__':
    run()
