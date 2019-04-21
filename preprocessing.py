import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Normalize a dataframe by fitting min/max normalization up until n_test timesteps, and transforming all data
def normalize(input_df, n_test):
    values = input_df.values.astype('float32')
    train_values = values[0:-n_test]
    test_values = values[-n_test:]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = np.vstack([scaler.fit_transform(train_values), scaler.transform(test_values)])
    return pd.DataFrame(scaled, columns=input_df.columns, index=input_df.index), scaler

# Denormalize both true and predicted values for a city
def denormalize(scaled, scaler, city, preds):
    y = pd.DataFrame(scaler.inverse_transform(scaled), columns=scaled.columns)[city]
    scaled[city] = preds
    yhat = pd.DataFrame(scaler.inverse_transform(scaled), columns=scaled.columns)[city]
    return y, yhat

def remove_zeros(df, n_test):
    n_train = len(df) - n_test
    to_delete = []
    for col in df.columns:
        if all(df[col].values[:n_train] == 0):
            to_delete.append(col)
    return df.drop(to_delete, axis=1)

# Convert time series format to format for supervised machine learning
# Adapted from machine learning mastery tutorial
def to_supervised(input_df, cities_in, cities_out, lags_in, lags_out, n_test):
    df = input_df.copy()
    # input sequence (t-n, ... t-1)
    xcols, ycols = [], []
    for city in cities_in:
        for i in lags_in:
            xcols.append(df[city].shift(i))
    x = pd.concat(xcols, axis=1)
    # forecast sequence (t, t+1, ... t+n)
    for city in cities_out:
        for i in lags_out:
            ycols.append(df[city].shift(-i))
    y = pd.concat(ycols, axis=1)
    df['date'] = df.index
    dates = pd.concat([df['date'].shift(-i) for i in lags_out], axis=1)
    dates.index = x.index
    # drop early rows that have nan for some predictors
    nullrows = x[(pd.isnull(x).any(axis=1))].index
    x, y, dates = x.drop(nullrows, axis=0), y.drop(nullrows, axis=0), dates.drop(nullrows, axis=0)
    # drop late rows tdhat have nan for some targets
    nullrows = y[(pd.isnull(y).any(axis=1))].index
    x, y, dates = x.drop(nullrows, axis=0), y.drop(nullrows, axis=0), dates.drop(nullrows, axis=0)
    # train-test split, return arrays in shape (n_samples, n_timesteps, n_features)
    n_train = x.shape[0] - n_test
    x_train = x[:n_train].values.reshape(n_train, len(lags_in), len(cities_in), order='F')
    y_train = y[:n_train].values.reshape(n_train, len(lags_out), len(cities_out), order='F')
    x_test = x[n_train:].values.reshape(n_test, len(lags_in), len(cities_in), order='F')
    y_test = y[n_train:].values.reshape(n_test, len(lags_out), len(cities_out), order='F')
    dates_train = dates[:n_train].values.flatten()
    dates_test = dates[n_train:].values.flatten()
    return x_train, y_train, x_test, y_test, dates_train, dates_test

