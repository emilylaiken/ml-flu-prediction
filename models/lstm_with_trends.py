import json
import time
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from vis.visualization import visualize_saliency, visualize_cam
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mode

from preprocessing import normalize, denormalize, to_supervised
from utils import load_flu, load_dengue, load_flu_states, load_flu_cities_subset, remove_zeros
#from evaluate_models import plot_violins

def get_correlations(x_train, y_train):
    correlations = np.zeros((x_train.shape[1], x_train.shape[2]))
    for lag in range(x_train.shape[1]):
        for c in range(x_train.shape[2]):
            timeseries = x_train[:, lag, c]
            correlations[lag][c] = np.corrcoef(y_train, timeseries)[0][1]
    return correlations

# LSTM model
def lstm_with_trends(df, df_trends, th, n_test, long_test=False, labels=None):
    np.random.seed(0)
    normalized_df, scaler = normalize(df, n_test)
    x_train, y_train, x_test, y_test, dates_train, dates_test = to_supervised(normalized_df, df.columns, df.columns, range(52, 0, -1), [th-1], n_test)
    print('EPI DATA SHAPE', x_train.shape)

    trends_train_full = []
    trends_test_full = []
    for c, city in enumerate(df.columns):
        trends_city = remove_zeros(df_trends[city], n_test)
        normalized_trends_city, _ = normalize(trends_city, n_test)
        _, trends_train, _, trends_test, _, _ = to_supervised(normalized_trends_city, normalized_trends_city.columns[:1], normalized_trends_city.columns, range(52, 0, -1), [th-1], n_test)
        trends_train_full.append(trends_train)
        trends_test_full.append(trends_test)
    trends_train = np.concatenate(trends_train_full, axis=2)
    trends_test = np.concatenate(trends_test_full, axis=2)
    correlations = {}
    for c in range(len(df.columns)):
        print(c)
        for t in range(trends_train.shape[2]):
            corr = np.corrcoef(trends_train[:, :, t].flatten(), y_train[:, :, c].flatten())[0][1]
            if str(corr) != 'nan':
                if t in correlations.keys():
                    correlations[t] = max([correlations[t], corr])
                else:
                    correlations[t] = corr
    best_trends = sorted(correlations, key=correlations.get, reverse=True)[:10]
    print({t: correlations[t] for t in best_trends})
    trends_train = trends_train[:, :, best_trends]
    trends_test = trends_test[:, :, best_trends]
    trends_train = np.hstack((np.full((trends_train.shape[0], th-1, trends_train.shape[2]), -1), trends_train))
    trends_test = np.hstack((np.full((trends_test.shape[0], th-1, trends_test.shape[2]), -1), trends_test))
    print('TRENDS DATA SHAPE', trends_train.shape)

    x_train_ext = np.hstack((x_train, np.full((x_train.shape[0], trends_train.shape[1], x_train.shape[2]), -1)))
    x_test_ext = np.hstack((x_test, np.full((x_test.shape[0], trends_test.shape[1], x_test.shape[2]), -1)))
    trends_train_ext = np.hstack((np.full((x_train.shape[0], x_train.shape[1], trends_train.shape[2]), -1), trends_train))
    trends_test_ext = np.hstack((np.full((x_test.shape[0], x_test.shape[1], trends_test.shape[2]), -1), trends_test))
    x_train = np.concatenate((x_train_ext, trends_train_ext), axis=2)
    x_test = np.concatenate((x_test_ext, trends_test_ext), axis=2)
    print('TOGETHER DATA SHAPE', x_train.shape)

    y_train = y_train.reshape(y_train.shape[0]*y_train.shape[1], y_train.shape[2])
    y_test = y_test.reshape(y_test.shape[0]*y_test.shape[1], y_test.shape[2])

    if not long_test:
        x_test, y_test, dates_test = x_test[0:1], y_test[0:1], dates_test[0:1]

    
    # design network
    def init_net(nodes):
        model = Sequential()
        #model.add(Dropout(0.9, input_shape=(x_train.shape[1], x_train.shape[2])))
        model.add(GRU(best_nodes, input_shape=(x_train.shape[1], x_train.shape[2]), dropout=0.3))
        model.add(Dense(y_train.shape[1]))
        model.compile(loss='mse', optimizer=Adam(lr=0.001))
        return model
    best_nodes, best_epochs = 5, 1000
    model = init_net(best_nodes)
    history = model.fit(x_train, y_train, epochs=best_epochs, batch_size=32, validation_data=(x_test, y_test), verbose=1, shuffle=False)
    labels = df.columns
    yhat_train_all = model.predict(x_train)
    yhat_test_all = model.predict(x_test)
    coefs = {city:{} for city in df.columns}

    preds = {}
    for c in range(yhat_train_all.shape[1]):
        city = df.columns[c]
        # Un-scale true values and predictions
        y_train, yhat_train = denormalize(normalized_df.loc[dates_train], scaler, city, yhat_train_all[:, c])
        y_test, yhat_test = denormalize(normalized_df.loc[dates_test], scaler, city, yhat_test_all[:, c])
        #preds[city] = ((dates_train, dates_test), (y_train, y_test), (yhat_train, yhat_test))
        preds[city] = ([str(x) for x in list(dates_test)], list(y_test.values), list(yhat_test.values))
    return preds, coefs
