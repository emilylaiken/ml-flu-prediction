import json
import numpy as np
import pandas as pd

from preprocessing import normalize, denormalize
from utils import load_flu, load_dengue, load_flu_states, load_flu_cities_subset

# Persistance model
def persistance(df, th, n_test, long_test=False): 
    normalized_df, scaler = normalize(df, n_test)
    n_train = len(normalized_df) - n_test
    preds = {}
    for city in df.columns:
        y_train, yhat_train = normalized_df[:n_train][city].values[th:], normalized_df[city].shift(th)[:n_train].values[th:]
        if long_test:
            y_test, yhat_test = normalized_df[n_train:][city].values, normalized_df[city].shift(th)[n_train:].values
            dates_train, dates_test = normalized_df[:n_train].index.values[th:], normalized_df[n_train:].index.values
        else:
            y_test, yhat_test = normalized_df[n_train:n_train+1][city].values, normalized_df[city].shift(th)[n_train:n_train+1].values
            dates_train, dates_test = normalized_df[:n_train].index.values[th:], normalized_df[n_train:n_train+1].index.values
        # Un-scale true values and predictions
        y_train, yhat_train = denormalize(normalized_df.loc[dates_train], scaler, city, yhat_train)
        y_test, yhat_test = denormalize(normalized_df.loc[dates_test], scaler, city, yhat_test)
        # Return 
        #preds[city] = ((dates_train, dates_test), (y_train.values, y_test.values), (yhat_train.values, yhat_test.values))
        preds[city] = ([str(x) for x in list(dates_test)], list(y_test.values), list(yhat_test.values))
    return preds, {city:{} for city in df.columns}
