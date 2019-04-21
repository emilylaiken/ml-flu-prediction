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
from evaluate_models import plot_violins
#from evaluate_models import plot_violins

# LSTM model
def lstm_with_trends(df, df_trends, th, n_test, long_test=False):
    #K.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)))
    np.random.seed(0)
    normalized_df, scaler = normalize(df, n_test)
    trends = pd.concat([df_trends[city] for city in df.columns], axis=1)
    trends = remove_zeros(trends, n_test)
    normalized_trends, _ = normalize(trends, n_test)
    _, trends_train, _, trends_test, _, _ = to_supervised(normalized_trends, normalized_trends.columns[:1], normalized_trends.columns, range(52, 0, -1), [th-1], n_test)
    x_train, y_train, x_test, y_test, dates_train, dates_test = to_supervised(normalized_df, df.columns, df.columns, range(52, 0, -1), [th-1], n_test)
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
    # design network
    def init_net(nodes):
        model = Sequential()
        model.add(GRU(best_nodes, input_shape=(x_train.shape[1], x_train.shape[2]), dropout=0.3))
        model.add(Dense(y_train.shape[1]))
        model.compile(loss='mse', optimizer=Adam(lr=0.001))
        return model
    best_nodes, best_epochs = 5, 2
    model = init_net(best_nodes)
    history = model.fit(x_train, y_train, epochs=best_epochs, batch_size=64, validation_data=(x_test, y_test), verbose=1, shuffle=False)
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
