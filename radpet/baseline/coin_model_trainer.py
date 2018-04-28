import os
import numpy as np
import pandas as pd
from keras import Sequential
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, GRU

from radpet.utilki.metrics import mape
from radpet.utilki.preprocess import scale, inverse_scale
from radpet.utilki.timeframe import series_to_supervised


def get_features(data):
    features = pd.DataFrame()
    features['price'] = data['price']
    # features['marketCap'] = data['marketCap']
    return features


def preprocess(data):
    scaled, scaler = scale(data)
    data = series_to_supervised(scaled, 1, 1)
    # data = data.drop(data.columns[[3]], axis=1)

    return data, scaler


def split(data):
    train_X, train_y = data[:, :-1], data[:, -1]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    return train_X, train_y


def create_model(timesteps, units):
    model = Sequential()
    model.add(GRU(100, input_shape=(timesteps, units)))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam', metrics=['mse'])

    model.summary()

    return model


def create_model_for_coin(coin, basepath='./only_price_on_full'):
    coin_data = pd.read_csv('../data/train/' + str(coin) + '.csv', index_col='time')
    coin_data_test = pd.read_csv('../data/test/' + str(coin) + '.csv', index_col='time')

    features = get_features(coin_data).values
    features, scaler = preprocess(features)

    # n_train = 10000
    #
    # train = features.values[:n_train, :]
    # test = features.values[n_train:, :]
    #
    # train_X, train_y = split(train)
    #
    # test_X, test_y = split(test)

    train_X, train_y = split(features.values)

    features_test = get_features(coin_data_test).values
    features_test, scaler_test = preprocess(features_test)

    test_X, test_y = split(features_test.values)

    # callbacks
    coin_path = os.path.join(basepath, str(coin))
    if not os.path.isdir(coin_path):
        os.mkdir(coin_path)
        os.mkdir(os.path.join(coin_path, 'checkpoints'))

    tensorboard = TensorBoard(log_dir=os.path.join(coin_path, 'log'))

    checkpoint = ModelCheckpoint(
        os.path.join(os.path.join(coin_path, 'checkpoints'), 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'),
        save_best_only=True)
    earlystopping = EarlyStopping(patience=4)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=3, min_lr=0.001)

    model = create_model(train_X.shape[1], train_X.shape[2])
    model.fit(train_X, train_y, epochs=20, batch_size=60, validation_data=(test_X, test_y),
              shuffle=False, callbacks=[tensorboard, checkpoint, earlystopping])

    yhat = model.predict(test_X)

    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

    inv_yhat = inverse_scale(yhat, test_X, scaler_test)

    test_y = test_y.reshape((len(test_y), 1))
    inv_y = inverse_scale(test_y, test_X, scaler_test)

    df_y_pred = pd.DataFrame(data={
        'price': inv_yhat
    })
    df_y_pred.to_csv(os.path.join(coin_path, 'checkpoints/y_pred_val.csv'), index=False)

    score_mape = mape(y_pred=inv_yhat, y_true=inv_y)
    print('Mape score on val {} for {} '.format(score_mape, coin))

    return score_mape


if __name__ == '__main__':

    selected_coins = pd.read_csv('../SELECTED_COINS.csv').values.reshape((20,))
    scores = []
    for coin in selected_coins:
        scores.append(create_model_for_coin(coin))
        # #
        # test = pd.read_csv('../data/test/1442.csv', index_col='time')
        # test = get_features(test).values
        # test_scaled, test_scaler = preprocess(test)
        #
        # test_X, test_y = split(test_scaled.values)
        #
        # yhat = model.predict(test_X)
        # test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
        #
        # inv_yhat = inverse_scale(yhat, test_X, scaler)
        #
        # test_y = test_y.reshape((len(test_y), 1))
        # inv_y = inverse_scale(test_y, test_X, scaler)
        #
        # score_mape = mape(y_pred=inv_yhat, y_true=inv_y)
        # print('Mape score on test ', score_mape)

    print(np.array(scores).mean())
