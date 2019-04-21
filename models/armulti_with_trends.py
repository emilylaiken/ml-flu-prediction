import json
import multiprocessing
from pathos.multiprocessing import ProcessingPool as Pool
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

from preprocessing import normalize, denormalize, to_supervised, remove_zeros
from utils import load_flu, load_dengue, load_flu_states, load_trends_states, load_trends_cities, load_flu_cities_subset
#from evaluate_models import plot_violins

def get_correlations(x_train, y_train):
    correlations = np.zeros((x_train.shape[1], x_train.shape[2]))
    for lag in range(x_train.shape[1]):
        for c in range(x_train.shape[2]):
            timeseries = x_train[:, lag, c]
            correlations[lag][c] = np.corrcoef(y_train, timeseries)[0][1]
    return correlations


# AR-multi model
def ar_multi_with_trends(df, df_trends, th, n_test, nonlinear=False, long_test=False):
    preds = {}
    normalized_df, scaler = normalize(df, n_test)
    def calc_city(city):
        print(city)
        trends = pd.concat([df_trends[city] for city in df.columns], axis=1)
        trends = remove_zeros(trends, n_test)
        normalized_trends, _ = normalize(trends, n_test)
        _, trends_train, _, trends_test, _, _ = to_supervised(normalized_trends, normalized_trends.columns[:1], normalized_trends.columns, range(52, 0, -1), [th-1], n_test)
        x_train, y_train, x_test, y_test, dates_train, dates_test = to_supervised(normalized_df, df.columns, [city], range(52, 0, -1), [th-1], n_test)
        if not long_test:
            x_test, y_test, dates_test, trends_test = x_test[0:1], y_test[0:1], dates_test[0:1], trends_test[0:1]
        # Reshape y data to flatten timesteps
        y_train = y_train.reshape(y_train.shape[0], y_train.shape[1]*y_train.shape[2]).flatten()
        y_test = y_test.reshape(y_test.shape[0], y_test.shape[1]*y_test.shape[2]).flatten()
        # Select best 30 predictors from lagged timeseries for all cities
        correlations = get_correlations(x_train, y_train)
        correlations_trends = get_correlations(trends_train, y_train)
        scores = {}
        if nonlinear:
            alphas = [2, 4, 8, 16]
        else:
            alphas = [1e-5, 1e-4, 1e-03, 1e-02, 1e-01]
        for n_series in [10, 20, 40]:
            for n_series_trends in [10, 20, 40]:
                for alpha in alphas:
                    top_series = np.unravel_index(np.argsort(correlations.ravel())[-n_series:], correlations.shape)
                    top_trends = np.unravel_index(np.argsort(correlations_trends.ravel())[-n_series_trends:], correlations_trends.shape)
                    x_train_pruned = np.vstack([x_train[:, top_series[0][i], top_series[1][i]] for i in range(n_series)]).T
                    trends_train_pruned = np.vstack([trends_train[:, top_trends[0][i], top_trends[1][i]] for i in range(n_series_trends)]).T
                    x_train_pruned = np.hstack([x_train_pruned, trends_train_pruned])
                    if nonlinear:
                        lr = RandomForestRegressor(n_estimators=50, max_depth=alpha)
                    else:
                        lr = Lasso(max_iter=100000, alpha=alpha)
                    scores[(n_series, n_series_trends, alpha)] = np.mean(cross_val_score(lr, x_train_pruned, y_train, scoring='neg_mean_squared_error', cv=4))
        best = max(scores.items(), key=lambda x: x[1]) 
        best_n_series, best_n_trends, best_alpha = best[0][0], best[0][1], best[0][2]
        top_series = np.unravel_index(np.argsort(correlations.ravel())[-best_n_series:], correlations.shape)
        top_trends = np.unravel_index(np.argsort(correlations_trends.ravel())[-best_n_trends:], correlations_trends.shape)
        x_train = np.vstack([x_train[:, top_series[0][i], top_series[1][i]] for i in range(best_n_series)]).T
        trends_train = np.vstack([trends_train[:, top_trends[0][i], top_trends[1][i]] for i in range(best_n_trends)]).T
        x_train = np.hstack([x_train, trends_train])
        x_test = np.vstack([x_test[:, top_series[0][i], top_series[1][i]] for i in range(best_n_series)]).T
        trends_test = np.vstack([trends_test[:, top_trends[0][i], top_trends[1][i]] for i in range(best_n_trends)]).T
        x_test = np.hstack([x_test, trends_test])
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
        for j in range(best_n_trends):
            if coef_vector[best_n_series + j] != 0:
                feature_importances['term_' + trends.columns[top_trends[1][j]]] = coef_vector[best_n_series + j].item()
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
