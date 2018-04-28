import os

import pandas as pd
from keras import Sequential
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, GRU

from radpet.utilki.metrics import mape
from radpet.utilki.preprocess import scale, inverse_scale
from radpet.utilki.timeframe import series_to_supervised

LABEL = 'price'


def run(data):
    pass


def get_features(data):
    features = pd.DataFrame()
    features['price'] = data['price']
    features['marketCap'] = data['marketCap']
    features['supply'] = data['CirculatingSupply']
    # features['volume24h'] = data['Volume24h']
    return features


def preprocess(data):
    scaled, scaler = scale(data)
    data = series_to_supervised(scaled, 1, 1)
    data = data.drop(data.columns[[4, 5]], axis=1)

    return data, scaler


def split(data):
    train_X, train_y = data[:, :-1], data[:, -1]
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    return train_X, train_y


if __name__ == '__main__':
    btc = pd.read_csv('../data/train/1442.csv', index_col='time')
    print(btc.head())
    features = get_features(btc).values
    features, scaler = preprocess(features)

    n_train = 10000

    train = features.values[:n_train, :]
    test = features.values[n_train:, :]

    train_X, train_y = split(train)

    test_X, test_y = split(test)

    model = Sequential()
    model.add(GRU(100, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam', metrics=['mse'])

    model.summary()

    # callbacks

    tensorboard = TensorBoard(log_dir='./logs')
    checkpoint = ModelCheckpoint(os.path.join('./checkpoints', 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'))
    earlystopping = EarlyStopping(patience=4)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                  patience=3, min_lr=0.001)

    model.fit(train_X, train_y, epochs=20, batch_size=60, validation_data=(test_X, test_y),
              shuffle=False, callbacks=[tensorboard, checkpoint, earlystopping])

    yhat = model.predict(test_X)

    test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

    inv_yhat = inverse_scale(yhat, test_X, scaler)

    test_y = test_y.reshape((len(test_y), 1))
    inv_y = inverse_scale(test_y, test_X, scaler)

    df_y_pred = pd.DataFrame(data={
        'price': inv_yhat
    })
    df_y_pred.to_csv('./checkpoints/y_pred_val.csv', index=False)

    score_mape = mape(y_pred=inv_yhat, y_true=inv_y)
    print('Mape score on val ', score_mape)
    #
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
