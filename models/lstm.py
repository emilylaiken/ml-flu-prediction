import json
import time
import sys
import shap
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

from preprocessing import normalize, denormalize, to_supervised
from utils import load_flu, load_dengue, load_flu_states, load_flu_cities_subset
#from evaluate_models import plot_violins

# LSTM model
def lstm(df, th, n_test, long_test=False, labels=None):
    np.random.seed(0)
    normalized_df, scaler = normalize(df, n_test)
    x_train, y_train, x_test, y_test, dates_train, dates_test = to_supervised(normalized_df, df.columns, df.columns, range(52, 0, -1), [th-1], n_test)
    if not long_test:
        x_test, y_test, dates_test = x_test[0:1], y_test[0:1], dates_test[0:1]
    y_train = y_train.reshape(y_train.shape[0]*y_train.shape[1], y_train.shape[2])
    y_test = y_test.reshape(y_test.shape[0]*y_test.shape[1], y_test.shape[2])
    print(x_test[0].shape)
    # design network
    def init_net(nodes):
        model = Sequential()
        model.add(GRU(best_nodes, input_shape=(x_train.shape[1], x_train.shape[2]), dropout=0.3))
        #model.add(LSTM(best_nodes, input_shape=(x_train.shape[1], x_train.shape[2])))
        model.add(Dense(x_train.shape[2]))
        model.compile(loss='mse', optimizer=Adam(lr=0.001))
        return model
    # cross validation to determine # of nodes
    #kf = KFold(n_splits=4)
    #scores = {}
    #for nodes in [5, 10, 20]:
    #    for epochs in [10, 20, 40]:
    #        model = init_net(nodes)
    #        score = []
    #        for train_index, test_index in kf.split(x_train):
    #            x_train_split, x_test_split = x_train[train_index], x_train[test_index]
    #            y_train_split, y_test_split = y_train[train_index], y_train[test_index]
    #            model.fit(x_train_split, y_train_split, epochs=epochs, batch_size=64, verbose=0, shuffle=False)
    #            score.append(-model.evaluate(x_test_split, y_test_split))
    #        scores[(nodes, epochs)] = np.mean(score)
    # fit network
    #best = max(scores.items(), key=lambda x: x[1])
    #best_nodes, best_epochs = best[0][0], best[0][1]
    best_nodes, best_epochs = 5, 150
    model = init_net(best_nodes)
    #earlystop = EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
    history = model.fit(x_train, y_train, epochs=best_epochs, batch_size=32, validation_data=(x_test, y_test), verbose=1, shuffle=False)
    #print(model.layers[1].get_weights()[0].shape, model.layers[1].get_weights()[1].shape)
    # predict on test data
    yhat_train_all = model.predict(x_train)
    yhat_test_all = model.predict(x_test)
    '''
    weights = model.layers[-1].get_weights()
    for c in range(len(labels)):
        second_term = [0 for _ in range(len(weights[-1]))]
        second_term[0] = weights[-1][c]
        first_term = [[0 for _ in range(len(weights[-1]))] for _ in range(5)]
        for w in range(len(first_term)):
            first_term[w][0] = weights[0][w][c]
        model.layers[-1].set_weights([np.array(first_term), np.array(second_term)])
        saliency_pos = visualize_saliency(model, layer_idx=-1, filter_indices=0, seed_input=x_test[0], grad_modifier=None, keepdims=True)
        saliency_neg = visualize_saliency(model, layer_idx=-1, filter_indices=0, seed_input=x_test[0], grad_modifier='negate', keepdims=True)
        saliency_small = visualize_saliency(model, layer_idx=-1, filter_indices=0, seed_input=x_test[0], grad_modifier='small_values', keepdims=True)
        for label, saliency, cmap in [('Positive', saliency_pos, 'Reds'), ('Negative', saliency_neg, 'Blues'), ('Small', saliency_small, 'Greens')]:
            fig, ax = plt.subplots(1, figsize=(20, 15))
            sns.heatmap(saliency, ax=ax, cmap=cmap, vmin=0, vmax=1, yticklabels=[x if x % 4 == 0 else '' for x in range (52, 0, -1)])
            plt.xticks(np.arange(len(labels)), labels, rotation='vertical')
            ax.set_title(label + ' Gradients for GRU Predictions of ILI in 37 States for Week of March 6, 2016', fontsize='x-large')
            plt.tight_layout()
            plt.savefig('saliency/by_state/' + labels[c] + '_saliency' + label + '.png')
    '''
    preds = {}
    for c in range(yhat_train_all.shape[1]):
        city = df.columns[c]
        # Un-scale true values and predictions
        y_train, yhat_train = denormalize(normalized_df.loc[dates_train], scaler, city, yhat_train_all[:, c])
        y_test, yhat_test = denormalize(normalized_df.loc[dates_test], scaler, city, yhat_test_all[:, c])
        #preds[city] = ((dates_train, dates_test), (y_train, y_test), (yhat_train, yhat_test))
        preds[city] = ([str(x) for x in list(dates_test)], list(y_test.values), list(yhat_test.values))
    return preds, {city:{} for city in df.columns}
