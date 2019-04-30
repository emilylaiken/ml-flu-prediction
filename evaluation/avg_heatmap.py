import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.gridspec import GridSpec
import seaborn as sns
import matplotlib.dates as mdates

from evaluate_models import plot_lines_single
sys.path.insert(0, '../')
from utils import load_flu, load_flu_states, load_run

geogran = 'state' # Either 'state' or 'city'
th = 2 # 1, 2, 4, or 8
loc = 'OR' # Name of city/state
model = 'gru'
fname = 'bla21.png'

# Load data, set general parameters
if geogran == 'state':
    df = load_flu_states('../')
else:
    df = load_flu_cities_subset('../')
#with open('../results/avg_heatmaps/' + geogran + '/' + str(th) + '/' + model + '/' + loc + '.json') as infile:
#	coefs = json.load(infile)
data = np.load('../results/avg_heatmaps/city/8/gru/city2_pos.npy')

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
years = mdates.YearLocator()
years_format = mdates.DateFormatter('%Y')

fig, ax = plt.subplots(1, figsize=(20, 20))
xticklabels = df.columns
#data = np.full((52, len(df.columns)), float(coefs['pos']['other']))
#for feature, val in coefs['pos'].items():
#	if feature != 'other':
#		data[int(feature.split('_')[-1])-1][list(df.columns).index(feature.split('_')[0])] = float(val)
sns.heatmap(data, ax=ax, cmap='RdBu_r', vmin=-1, vmax=1, cbar=False)
ax.set_xticklabels(xticklabels)
ax.set_xlabel('Location')
ax.set_ylabel('Lag')
plt.savefig(fname)