import multiprocessing
import json
from pathos.multiprocessing import ProcessingPool as Pool
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score

from preprocessing import normalize, denormalize, to_supervised, remove_zeros
from utils import load_flu, load_dengue, load_flu_states, load_trends_states, load_trends_cities, load_flu_cities_subset


# AR model
def ar_with_trends(df, df_trends, th, n_test, long_test=False):
    preds = {}
    normalized_df, scaler = normalize(df, n_test)
    def calc_city(city):
        trends = df_trends[city]
        trends = remove_zeros(trends, n_test)
        normalized_trends, _ = normalize(trends, n_test)
        _, trends_train, _, trends_test, _, _ = to_supervised(normalized_trends, normalized_trends.columns[:1], normalized_trends.columns, range(52, 0, -1), [th-1], n_test)
        x_train, y_train, x_test, y_test, dates_train, dates_test = to_supervised(normalized_df, [city], [city], range(52, 0, -1), [th-1], n_test)
        if not long_test:
            x_test, y_test, dates_test, trends_test = x_test[0:1], y_test[0:1], dates_test[0:1], trends_test[0:1]
        # Reshape x data and combine trends and epi data
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])
        trends_train = trends_train.reshape(trends_train.shape[0], trends_train.shape[1]*trends_train.shape[2])
        trends_test = trends_test.reshape(trends_test.shape[0], trends_test.shape[1]*trends_test.shape[2])
        x_train = np.hstack([x_train, trends_train])
        # Reshape y data to flatten time
        x_test = np.hstack([x_test, trends_test])
        y_train, y_test = y_train.flatten(), y_test.flatten()
        # Cross-validation to determine best alpha penalty parameter
        scores = {}
        for alpha in [1e-5, 1e-4, 1e-03, 1e-02, 1e-01]:
            lr = Lasso(max_iter=100000, alpha=alpha)
            scores[alpha] = np.mean(cross_val_score(lr, x_train, y_train, scoring='neg_mean_squared_error', cv=4))
        best_alpha = max(scores.items(), key=lambda x: x[1])[0] 
        lr = Lasso(max_iter=100000, alpha=best_alpha)
        lr.fit(x_train, y_train)
        feature_importances = {city + '_' + str(51-i): lr.coef_[i].item() for i in range(52) if lr.coef_[i] != 0}
        for j in range(len(trends.columns)):
            if lr.coef_[52 + j] != 0:
                feature_importances['term_' + trends.columns[j]] = lr.coef_[52 + j].item()
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

'''
df = load_flu_cities_subset()
df_trends = load_trends_cities()
model_name = 'nowcasting_ar'

print('# of CPUs:')
print(multiprocessing.cpu_count())
for th in [1, 2, 4, 8]:
    print('TH: ' + str(th))
    print('--------------')
    preds = {city:{'dates':[], 'ytrues':[], 'yhats':[], 'coefs':[]} for city in df.columns}
    for n_test in range(200, 0, -1):
        print(n_test)
        run, coefs = ar(df, df_trends, th, n_test)
        for city in df.columns:
            preds[city]['dates'].append(run[city][0][0])
            preds[city]['ytrues'].append(run[city][1][0])
            preds[city]['yhats'].append(run[city][2][0])
            preds[city]['coefs'].append(coefs[city])
    with open('results_cities/' + model_name + '/' + str(th) + '.json' , 'w') as outfile:
        json.dump(preds, outfile)
'''



