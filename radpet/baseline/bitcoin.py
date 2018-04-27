import os

import numpy as np
import pandas as pd
from keras import Sequential
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

from radpet.utilki.timeframe import series_to_supervised
from radpet.utilki.metrics import mape

LABEL = 'price'


def run(data):
    pass


def get_features(data):
    features = pd.DataFrame()
    features['price'] = data['price']
    # features['marketCap'] = data['marketCap']
    # features['supply'] = data['CirculatingSupply']
    # features['volume24h'] = data['Volume24h']
    return features


def preprocess(data):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data)
    data = series_to_supervised(scaled, 1, 1)
    return data, scaler


def split(data):
    values = data.values
    n_train = 10000

    train = values[:n_train, :]
    test = values[n_train:, :]

    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]

    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    return train_X, train_y, test_X, test_y


if __name__ == '__main__':
    btc = pd.read_csv('../data/train/1442.csv', index_col='time')
    print(btc.head())
    features = get_features(btc).values
    features, scaler = preprocess(features)
    train_X, train_y, test_X, test_y = split(features)

    model = Sequential()
    model.add(LSTM(100, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam', metrics=['mse'])

    model.summary()

    # callbacks

    tensorboard = TensorBoard(log_dir='./logs')
    checkpoint = ModelCheckpoint(os.path.join('./checkpoints', 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'),
                                 monitor='mse')

    model.fit(train_X, train_y, epochs=60, batch_size=78, validation_data=(test_X, test_y),
              shuffle=False, callbacks=[tensorboard, checkpoint])

    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

    inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]

    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]

    score_mape = mape(y_pred=inv_yhat, y_true=inv_y)
    print('Mape score on val ', score_mape)

    test = pd.read_csv('../data/test/1442.csv', index_col='time')
    test = get_features(test)
    features_test, scaler = preprocess(test)
    features_test = features_test.values
    test_X, test_y = features_test[:, :-1], features_test[:, -1]
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

    inv_yhat = np.concatenate((yhat, test_X[:, 1:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]

    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, test_X[:, 1:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]

    score_mape = mape(y_pred=inv_yhat, y_true=inv_y)
    print('Mape score on test ', score_mape)
