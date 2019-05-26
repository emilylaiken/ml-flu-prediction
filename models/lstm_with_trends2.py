import json
import sys
import time
import multiprocessing
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, Masking
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
import matplotlib.pyplot as plt

from preprocessing import normalize, denormalize, to_supervised, remove_zeros
from utils import load_flu, load_dengue, load_flu_states, load_trends_states, load_trends_cities, load_flu_cities_subset


def get_correlations(x_train, y_train):
    correlations = np.zeros((x_train.shape[1], x_train.shape[2]))
    for lag in range(x_train.shape[1]):
        for c in range(x_train.shape[2]):
            timeseries = x_train[:, lag, c]
            correlations[lag][c] = np.corrcoef(y_train, timeseries)[0][1]
    return correlations

# LSTM model
def lstm_with_trends(df, df_trends, th, n_test, long_test=False):
    #K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)))
    np.random.seed(0)
    normalized_df, scaler = normalize(df, n_test)
    x_train, y_train, x_test, y_test, dates_train, dates_test = to_supervised(normalized_df, df.columns, df.columns, range(52, 0, -1), [th-1], n_test)
    print('SERIES ', x_train.shape)
    trends_train_full = []
    trends_test_full = []
    for c, city in enumerate(df.columns):
        trends_city = remove_zeros(df_trends[city], n_test)
        if len(trends_city.columns) > 10:
            n_trends = 10
        else:
            n_trends = len(trends_city.columns)
        normalized_trends_city, _ = normalize(trends_city, n_test)
        _, trends_train, _, trends_test, _, _ = to_supervised(normalized_trends_city, normalized_trends_city.columns[:1], normalized_trends_city.columns, range(52, 0, -1), [th-1], n_test)
        correlations = get_correlations(trends_train, y_train[:, 0, c])
        top_trends = np.unravel_index(np.argsort(correlations.ravel())[-n_trends:], correlations.shape)
        trends_train_pruned = np.concatenate([trends_train[:, top_trends[0][i], top_trends[1][i]].reshape(len(trends_train), 1, 1) for i in range(n_trends)], axis=2)
        trends_test_pruned = np.concatenate([trends_test[:, top_trends[0][i], top_trends[1][i]].reshape(len(trends_test), 1, 1) for i in range(n_trends)], axis=2)
        trends_train_full.append(trends_train_pruned)
        trends_test_full.append(trends_test_pruned)
    trends_train = np.concatenate(trends_train_full, axis=2)
    trends_test = np.concatenate(trends_test_full, axis=2)
    #trends_train = trends_train.reshape((trends_train.shape[0], 1, trends_train.shape[1]), order='F')
    #trends_test = trends_test.reshape((trends_test.shape[0], 1, trends_test.shape[1]), order='F')
    print('TRENDS ', trends_train.shape)
    #_, trends_train, _, trends_test, _, _ = to_supervised(normalized_trends, normalized_trends.columns[:1], normalized_trends.columns, range(52, 0, -1), [th-1], n_test)
    trends_train = np.hstack((np.full((trends_train.shape[0], th-1, trends_train.shape[2]), -1), trends_train))
    trends_test = np.hstack((np.full((trends_test.shape[0], th-1, trends_test.shape[2]), -1), trends_test))
    if not long_test:
        x_test, y_test, dates_test, trends_test = x_test[0:1], y_test[0:1], dates_test[0:1], trends_test[0:1]
    # Stack x data together (trends and epi data)
    x_train_ext = np.hstack((x_train, np.full((x_train.shape[0], trends_train.shape[1], x_train.shape[2]), -1))) # Pad with -1
    trends_train_ext = np.hstack((np.full((x_train.shape[0], x_train.shape[1], trends_train.shape[2]), -1), trends_train))
    x_train = np.concatenate((x_train_ext, trends_train_ext), axis=2)
    x_test_ext = np.hstack((x_test, np.full((x_test.shape[0], trends_test.shape[1], x_test.shape[2]), -1))) # Pad with -1
    trends_test_ext = np.hstack((np.full((x_test.shape[0], x_test.shape[1], trends_test.shape[2]), -1), trends_test))
    x_test = np.concatenate((x_test_ext, trends_test_ext), axis=2)
    # Reshape y data
    y_train = y_train.reshape(y_train.shape[0]*y_train.shape[1], y_train.shape[2])
    y_test = y_test.reshape(y_test.shape[0]*y_test.shape[1], y_test.shape[2])
    print('TOGETHER', x_train.shape)
    # design network
    def init_net(nodes):
        model = Sequential()
        model.add(GRU(best_nodes, input_shape=(x_train.shape[1], x_train.shape[2]), dropout=0.3))
        model.add(Dense(y_train.shape[1]))
        model.compile(loss='mse', optimizer=Adam(lr=0.001))
        return model
    best_nodes, best_epochs = 5, 1000
    model = init_net(best_nodes)
    history = model.fit(x_train, y_train, epochs=best_epochs, batch_size=32, validation_data=(x_test, y_test), verbose=0, shuffle=False)
    # predict on test data
    yhat_train_all = model.predict(x_train)
    yhat_test_all = model.predict(x_test)
    preds = {}
    for c in range(yhat_train_all.shape[1]):
        city = df.columns[c]
        # Un-scale true values and predictions
        y_train, yhat_train = denormalize(normalized_df.loc[dates_train], scaler, city, yhat_train_all[:, c])
        y_test, yhat_test = denormalize(normalized_df.loc[dates_test], scaler, city, yhat_test_all[:, c])
        #preds[city] = ((dates_train, dates_test), (y_train, y_test), (yhat_train, yhat_test))
        preds[city] = ([str(x) for x in list(dates_test)], list(y_test.values), list(yhat_test.values))
    return preds, {city:{} for city in df.columns}
