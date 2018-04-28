import os

import numpy as np
import pandas as pd
from keras.models import load_model

from radpet.baseline.coin_model_trainer import get_features, preprocess, split
from radpet.utilki.metrics import mape
from radpet.utilki.preprocess import inverse_scale


def predict_for_coin(coin):
    test_data = pd.read_csv('./data/test/' + str(coin) + '.csv')
    score_mape = None
    for subdir, dirs, files in os.walk('./baseline/' + str(coin) + '/checkpoints'):
        files.sort()
        model = load_model(os.path.join(subdir, files[-2]))
        model.summary()

        features = get_features(test_data).values
        features, scaler = preprocess(features)

        test_X, test_y = split(features.values)

        yhat = model.predict(test_X)

        test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

        inv_yhat = inverse_scale(yhat, test_X, scaler)

        test_y = test_y.reshape((len(test_y), 1))
        inv_y = inverse_scale(test_y, test_X, scaler)

        df_y_pred = pd.DataFrame(data={
            'price': inv_yhat
        })
        df_y_pred.to_csv(os.path.join('./baseline/' + str(coin), 'y_pred_test.csv'), index=False)

        score_mape = mape(y_pred=inv_yhat, y_true=inv_y)
        print('Mape score on test set {} for {} '.format(score_mape, coin))

    return score_mape


def run():
    coins = pd.read_csv('./SELECTED_COINS.csv').values.reshape((20,))

    scores = []
    for coin in coins:
        score = predict_for_coin(coin=coin)
        scores.append(score)

    print(np.array(scores).mean())


if __name__ == '__main__':
    run()
