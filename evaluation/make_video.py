import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation

from utils import load_flu, load_flu_states, load_run

df = load_flu_states()
th = 1
max_features = 10
city = 'NY'
model = 'forecasting_rf'

print(df.columns)

# Load run
run = load_run('results_states/' + model + '/' + str(th) + '.json')[city]
dates, ytrues, yhats, coefs = [pd.to_datetime(date) for date in run['dates']], run['ytrues'], run['yhats'], run['coefs']

# Set up plotting space
fig, ax = plt.subplots(2, figsize=(20, 10))
plt.suptitle('Forecasting Influenza in ' + city + ' ' + str(th) + ' weeks ahead with linear ARGO-net', fontsize=20)

# Set-up heatmap
data = np.random.rand(max_features, len(df.columns))
im = ax[1].imshow(data, aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=0.5)
fig.colorbar(im, ax=ax, label='Magnitude of coefficient')
ax[1].set_title('Feature Importances')
ax[1].set_yticks(np.arange(max_features))
ax[1].set_yticklabels(range(1, max_features+1))
ax[1].set_xticks(np.arange(len(df.columns)))
ax[1].set_xticklabels(df.columns)
ax[1].set_xlabel('Location')
ax[1].set_ylabel('Lag')

# Set-up line graph
true_line, = ax[0].plot(dates, ytrues, color='grey', label='True Incidence')
model_line, = ax[0].plot(dates, yhats, color='green', label='Projected Incidence')
ax[0].legend(loc='upper right')
ax[0].set_ylabel('ILI incidence')
ax[0].set_title('Model Predictions')

def init():
    im.set_data(np.zeros((max_features, len(df.columns))))
    true_line.set_data([], [])
    model_line.set_data([], [])

def animate(i):
    if i % 50 == 0:
    	print(i)
    data = np.zeros((max_features, len(df.columns)))
    for feature, val in coefs[i].items():
    	if int(feature[2:]) < max_features:
    		data[int(feature[2:])][list(df.columns).index(feature[:2])] = val
    im.set_data(data)
    if i >= th:
        true_line.set_data(dates[:i-th], ytrues[:i-th])
    model_line.set_data(dates[:i], yhats[:i])
    return im, true_line, model_line,

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=200, repeat = False)
anim.save('videos/' + model + '/' + city + str(th) + '.mp4')

