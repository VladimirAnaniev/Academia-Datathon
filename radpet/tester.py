import pandas as pd
import os

from keras.models import load_model


def predict_for_coin(coin):
    test_data = pd.read_csv('./data/test/' + str(coin) + '.csv')
    print(test_data.head())


def run():
    coins = pd.read_csv('./SELECTED_COINS.csv').values.reshape((20,))[:1]
    for coin in coins:
        predict_for_coin(coin=coin)



if __name__ == '__main__':
    run()
