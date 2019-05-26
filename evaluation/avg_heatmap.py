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
models = ['lr', 'rf', 'gru']
fname = 'bla21.png'

# Load data, set general parameters
if geogran == 'state':
    df = load_flu_states('../')
else:
    df = load_flu_cities_subset('../')
#with open('../results/avg_heatmaps/' + geogran + '/' + str(th) + '/' + model + '/' + loc + '.json') as infile:
#	coefs = json.load(infile)
data = {}
for model in models:
	if model == 'gru':
		data[model + '_pos'] = np.load('../results/avg_heatmaps/city/8/' + model + '/AK_pos.npy')
		data[model + '_neg'] = np.load('../results/avg_heatmaps/city/8/' + model + '/AK_neg.npy')
	else:
		data[model = np.load('../results/avg_heatmaps/city/8/' + model + '/AK.npy')

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
years = mdates.YearLocator()
years_format = mdates.DateFormatter('%Y')

fig=plt.figure(figsize=(20, 20))
gs=GridSpec(5, 8, height_ratios=[1, 1, 1, 1, 1])
line_plot_ax=fig.add_subplot(gs[0,:7]) 
ar_heatmap_ax=fig.add_subplot(gs[0,7]) 
armulti_heatmap_ax=fig.add_subplot(gs[1,:]) 
rf_heatmap_ax=fig.add_subplot(gs[2,:])  
gru_pos_ax=fig.add_subplot(gs[3,:])
gru_neg_ax=fig.add_subplot(gs[4,:])    

fig, ax = plt.subplots(1, figsize=(20, 20))
sns.heatmap(data, ax=ax, cmap='RdBu_r', vmin=-1, vmax=1, cbar=False)
ax.set_xticklabels(xticklabels)
ax.set_xlabel('Location')
ax.set_ylabel('Lag')
plt.savefig(fname)



