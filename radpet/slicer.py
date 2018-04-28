import pandas as pd
import os


def run():
    coins = pd.read_csv('SELECTED_COINS.csv').values.reshape((20,))
    coins_data = pd.read_csv('./data/coins_filtered.csv')

    for coin in coins:
        coins_path = os.path.join('./data/subm_int', str(coin) + '.csv')
        coin_data = coins_data[coins_data['refID_coin'] == coin]
        coin_data.to_csv(coins_path, index='False')


if __name__ == '__main__':
    run()
