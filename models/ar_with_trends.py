import multiprocessing
import json
from pathos.multiprocessing import ProcessingPool as Pool
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score

from preprocessing import normalize, denormalize, to_supervised, remove_zeros
from utils import load_flu, load_dengue, load_flu_states, load_trends_states, load_trends_cities, load_flu_cities_subset

def get_correlations(x_train, y_train):
    correlations = np.zeros((x_train.shape[1], x_train.shape[2]))
    for lag in range(x_train.shape[1]):
        for c in range(x_train.shape[2]):
            timeseries = x_train[:, lag, c]
            correlations[lag][c] = np.corrcoef(y_train, timeseries)[0][1]
    return correlations

# AR model
def ar_with_trends(df, df_trends, th, n_test, long_test=False):
    preds = {}
    normalized_df, scaler = normalize(df, n_test)
    def calc_city(city):
        trends = df_trends[city]
        trends = remove_zeros(trends, n_test)
        print('N TRENDS: ', len(trends.columns))
        normalized_trends, _ = normalize(trends, n_test)
        _, trends_train, _, trends_test, _, _ = to_supervised(normalized_trends, normalized_trends.columns[:1], normalized_trends.columns, range(52, 0, -1), [th-1], n_test)
        x_train, y_train, x_test, y_test, dates_train, dates_test = to_supervised(normalized_df, [city], [city], range(52, 0, -1), [th-1], n_test)
        if not long_test:
            x_test, y_test, dates_test, trends_test = x_test[0:1], y_test[0:1], dates_test[0:1], trends_test[0:1]
        # Reshape x and y data to flatten time
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])
        y_train, y_test = y_train.flatten(), y_test.flatten()
        # Get N best time-series for trends
        correlations_trends = get_correlations(trends_train, y_train)
        scores = {}
        if len(trends.columns) < 10:
            trends_options = [len(trends.columns)]
        elif len(trends.columns) < 20:
            trends_options = [10]
        elif len(trends.columns) < 40:
            trends_options = [10, 20]
        else:
            trends_options = [10, 20, 40]
        alphas = [1e-5, 1e-4, 1e-03, 1e-02, 1e-01]
        for n_series_trends in trends_options:
            for alpha in alphas:
                top_trends = np.unravel_index(np.argsort(correlations_trends.ravel())[-n_series_trends:], correlations_trends.shape)
                trends_train_pruned = np.vstack([trends_train[:, top_trends[0][i], top_trends[1][i]] for i in range(n_series_trends)]).T
                x_train_pruned = np.hstack([x_train, trends_train_pruned])
                lr = Lasso(max_iter=100000, alpha=alpha)
                scores[(n_series_trends, alpha)] = np.mean(cross_val_score(lr, x_train_pruned, y_train, scoring='neg_mean_squared_error', cv=4))
        best = max(scores.items(), key=lambda x: x[1]) 
        best_n_trends, best_alpha = best[0][0], best[0][1]
        top_trends = np.unravel_index(np.argsort(correlations_trends.ravel())[-best_n_trends:], correlations_trends.shape)
        trends_train = np.vstack([trends_train[:, top_trends[0][i], top_trends[1][i]] for i in range(best_n_trends)]).T
        x_train = np.hstack([x_train, trends_train])
        trends_test = np.vstack([trends_test[:, top_trends[0][i], top_trends[1][i]] for i in range(best_n_trends)]).T
        x_test = np.hstack([x_test, trends_test])   
        print('TRAIN SHPAE: ', x_train.shape)     
        lr = Lasso(max_iter=100000, alpha=best_alpha)
        lr.fit(x_train, y_train)
        feature_importances = {city + '_' + str(51-i): lr.coef_[i].item() for i in range(52) if lr.coef_[i] != 0}
        for j in range(best_n_trends):
            if lr.coef_[52 + j] != 0:
                feature_importances['term_' + trends.columns[top_trends[1][j]]] = lr.coef_[52 + j].item()
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
