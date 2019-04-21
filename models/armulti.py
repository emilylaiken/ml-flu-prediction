import json
import time
import multiprocessing
from pathos.multiprocessing import ProcessingPool as Pool
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import normalize, denormalize, to_supervised
from utils import load_flu, load_flu_states, load_flu_cities_subset
from evaluate_models import plot_violins

# AR-multi model
def ar_multi(df, th, n_test, nonlinear=False, long_test=False):
    preds = {}
    normalized_df, scaler = normalize(df, n_test)
    def calc_city(city):
        print(city)
        x_train, y_train, x_test, y_test, dates_train, dates_test = to_supervised(normalized_df, df.columns, [city], range(52, 0, -1), [th-1], n_test)
        if not long_test:
            x_test, y_test, dates_test = x_test[0:1], y_test[0:1], dates_test[0:1]
        # Reshape y data to flatten timesteps
        y_train = y_train.reshape(y_train.shape[0], y_train.shape[1]*y_train.shape[2]).flatten()
        y_test = y_test.reshape(y_test.shape[0], y_test.shape[1]*y_test.shape[2]).flatten()
        # Select best 30 predictors from lagged timeseries for all cities
        correlations = np.zeros((x_train.shape[1], x_train.shape[2]))
        for lag in range(x_train.shape[1]):
            for c in range(x_train.shape[2]):
                timeseries = x_train[:, lag, c]
                correlations[lag][c] = np.corrcoef(y_train, timeseries)[0][1]
        scores = {}
        if nonlinear:
            alphas = [2, 4, 8, 16]
        else:
            alphas = [1e-5, 1e-4, 1e-03, 1e-02, 1e-01]
        for n_series in [10, 20, 40]:
            for alpha in alphas:
                top_series = np.unravel_index(np.argsort(correlations.ravel())[-n_series:], correlations.shape)
                x_train_pruned = np.vstack([x_train[:, top_series[0][i], top_series[1][i]] for i in range(n_series)]).T
                if nonlinear:
                    lr = RandomForestRegressor(n_estimators=50, max_depth=alpha)
                else:
                    lr = Lasso(max_iter=100000, alpha=alpha)
                scores[(n_series, alpha)] = np.mean(cross_val_score(lr, x_train_pruned, y_train, scoring='neg_mean_squared_error', cv=4))
        best = max(scores.items(), key=lambda x: x[1]) 
        best_n_series, best_alpha = best[0][0], best[0][1]
        top_series = np.unravel_index(np.argsort(correlations.ravel())[-best_n_series:], correlations.shape)
        x_train = np.vstack([x_train[:, top_series[0][i], top_series[1][i]] for i in range(best_n_series)]).T
        x_test = np.vstack([x_test[:, top_series[0][i], top_series[1][i]] for i in range(best_n_series)]).T
        if nonlinear:
            lr = RandomForestRegressor(n_estimators=50, max_depth=alpha)
        else:
            lr = Lasso(max_iter=100000, alpha=best_alpha)
        lr.fit(x_train, y_train)
        coefs = np.zeros((52, len(df.columns)))
        if nonlinear:
            coef_vector = lr.feature_importances_
        else:
            coef_vector = lr.coef_
        for i in range(best_n_series):
            coefs[51-top_series[0][i], top_series[1][i]] = coef_vector[i]
        feature_importances = {df.columns[j] + '_' + str(i): coefs[i][j] for i in range(coefs.shape[0]) for j in range(coefs.shape[1]) if coefs[i][j] != 0}
        yhat_train = lr.predict(x_train)
        yhat_test = lr.predict(x_test)
        # Un-scale true values and predictions
        y_train, yhat_train = denormalize(normalized_df.loc[dates_train], scaler, city, yhat_train)
        y_test, yhat_test = denormalize(normalized_df.loc[dates_test], scaler, city, yhat_test)
        return (city, ([str(x) for x in list(dates_test)], list(y_test.values), list(yhat_test.values)), feature_importances)
    pool = Pool(multiprocessing.cpu_count())
    results = pool.map(calc_city, df.columns)
    preds = {city: data for city, data, _ in results}
    coefs = {city: coef for city, _, coef in results}
    return preds, coefs


