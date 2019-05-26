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
mode = 'forecasting' # Either 'forecasting' or 'nowcasting'
th = 4 # 1, 2, 4, or 8
loc = 'GA' # Name of city/state
index = 138 # Index of date for heatmaps
#fname = 'heatmap_' + mode + '_' + str(th) + '_' + loc + '.png'
fname = 'figure4.png'

# Lags and states used for demo figures in paper:
# 1: PA, 142
# 2: ID, 105
# 4: ME, 130
# 8: KS, 63

# 1: NH, 77
# 2: MI, 130
# 4: NE, 74?
# 5: GA, 138

# Load data, set general parameters
if geogran == 'state':
    df = load_flu_states('../')
else:
    df = load_flu_cities_subset('../')
models = [mode + '_ar', mode + '_armulti', mode + '_rf', mode + '_gru']
#models = [mode + '_armulti', mode + '_rf', mode + '_gru']
max_features = 52
num_eval = int((len(df) - 52)*(5/10))
if mode == 'forecasting':
    label = 'AR'
else:
    label = 'ARGO'
labels = [label, label + '-net LR', label + '-net RF', label + '-net GRU (Positive Gradients)', label + '-net GRU (Negative Gradients)']
#labels = ['Linear Regression', 'Random Forest', 'Positive Gradients', 'Negative Gradients']
lineplotlabels = [label, label + '-net LR', label + '-net RF', label + '-net GRU', label + '-net GRU']
#lineplotlabels = ['Linear Regression', 'Random Forest', 'GRU']
colors = ['orange', 'mediumseagreen', 'royalblue', 'purple']

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
years = mdates.YearLocator()
years_format = mdates.DateFormatter('%Y')

# Load model predictions and feature importances 
runs = {}
for model in models:
    print(model)
    runs[model] = {}
    run = load_run('../results/' + geogran + '/' + str(th) + '/' + loc + '/' + model + '.json')
    runs[model]['dates'] = [pd.to_datetime(date) for date in run['dates']][-num_eval:]
    runs[model]['ytrues'] = run['ytrues'][-num_eval:]
    runs[model]['yhats'] = run['yhats'][-num_eval:]
    runs[model]['coefs'] = run['coefs'][-num_eval:]

# Create figure backbone
if mode == 'forecasting':
    fig=plt.figure(figsize=(20, 20))
    gs=GridSpec(5, 8, height_ratios=[1, 1, 1, 1, 1])
    #line_plot_ax=fig.add_subplot(gs[0,:]) 
    line_plot_ax=fig.add_subplot(gs[0,:7]) 
    ar_heatmap_ax=fig.add_subplot(gs[0,7]) 
    armulti_heatmap_ax=fig.add_subplot(gs[1,:]) 
    rf_heatmap_ax=fig.add_subplot(gs[2,:])  
    gru_pos_ax=fig.add_subplot(gs[3,:])
    gru_neg_ax=fig.add_subplot(gs[4,:])    
else:
    fig=plt.figure(figsize=(20, 20))
    gs=GridSpec(5, 5, height_ratios=[1, 1, 1, 1, 1])
    line_plot_ax=fig.add_subplot(gs[0,:3]) 
    ar_heatmap_ax=fig.add_subplot(gs[0,4]) 
    ar_gt_ax=fig.add_subplot(gs[0,3]) 
    armulti_heatmap_ax=fig.add_subplot(gs[2,:]) 
    armulti_gt_ax=fig.add_subplot(gs[1,:])
    rf_heatmap_ax=fig.add_subplot(gs[4,:])  
    rf_gt_ax=fig.add_subplot(gs[3,:])
    
# Line plot
if mode == 'forecasting':
    line_plot_ax.set_title(str(th) + '-week Forecasts (Historical Epi Data Only) for ILI in ' + loc, size='large')
    #line_plot_ax.set_title(str(th) + '-week Forecasts for ILI in ' + loc, size='x-large')
else:
    line_plot_ax.set_title(str(th) + '-week Nowcasts (Historical Epi Data and Real-Time GT Data) for ILI in ' + loc, size='large')
for i, model in enumerate(models):
    if i == 0:
        line_plot_ax.fill_between(runs[model]['dates'], runs[model]['ytrues'], color='lightgrey', label='Ground truth')
        line_plot_ax.set_ylim(0, max(runs[model]['ytrues']))
    line_plot_ax.plot(runs[model]['dates'], runs[model]['yhats'], label=lineplotlabels[i], color=colors[i], linewidth=3)
line_plot_ax.set_xlim(min(runs[models[0]]['dates']), max(runs[models[0]]['dates']))
line_plot_ax.xaxis.set_major_locator(years)
line_plot_ax.xaxis.set_major_formatter(years_format)
line_plot_ax.legend(loc='upper left')
line_plot_ax.spines['right'].set_visible(False)
line_plot_ax.spines['top'].set_visible(False)
line_plot_ax.axvline(runs[models[0]]['dates'][index], color='black', linewidth=4, dashes=[2, 2])
line_plot_ax.set_ylabel('ILI Incidence')

if mode == 'forecasting':
    lst = [(ar_heatmap_ax, None, models[0]), (armulti_heatmap_ax, None, models[1]), (rf_heatmap_ax, None, models[2]), (gru_pos_ax, None, models[3]), (gru_neg_ax, None, models[3])]
    #lst = [(armulti_heatmap_ax, None, models[0]), (rf_heatmap_ax, None, models[1]), (gru_pos_ax, None, models[2]), (gru_neg_ax, None, models[2])]
else:
    lst = [(ar_heatmap_ax, ar_gt_ax, models[0]), (armulti_heatmap_ax, armulti_gt_ax, models[1]), (rf_heatmap_ax, rf_gt_ax, models[2])]
for i, (ax, gt_ax, name) in enumerate(lst):
    model = runs[name]
    cmap = 'RdBu_r'
    if 'rf' in name:
        vmin, vmax = -0.5, 0.5
    else:
        vmin, vmax = -1, 1
    cbar = (i == 0)
    #cbar = False
    gt_terms = []
    if mode == 'nowcasting':
        for coef_lst in model['coefs']:
            for term in coef_lst.keys():
                if term[:4] == 'term' and term.split('_')[-1] not in gt_terms:
                    gt_terms.append(term.split('_')[-1])
    if 'armulti' in name or 'rf' in name:
        if 'armulti' in name:
            metric = 'Regression Coefficients'
        else:
            metric = 'Feature Importances'
        title = metric + ' in ' + labels[i] + ' for ' + model['dates'][index].strftime("%b. %d, %Y")
        titlesize = 'large'
        xticklabels = df.columns
        coefs = model['coefs'][index]
        data = np.zeros((max_features, len(df.columns)))
        terms = np.zeros((len(gt_terms), len(df.columns)))
        for feature, val in coefs.items():
            if feature != 'other' and feature[:4] != 'term':
                data[int(feature[2:])-1][list(df.columns).index(feature[:2])] = float(val)
            elif feature != 'other':
                terms[gt_terms.index(feature.split('_')[-1])][list(df.columns).index(feature.split('_')[1])] = float(val)
    elif 'gru' in name:
        vmin, vmax = 0, 1
        #metric = 'Saliency Map'
        #title = metric + ' for ' + labels[i] + ' for ' + model['dates'][index].strftime("%b. %d, %Y")
        title = labels[i]
        titlesize = 'small'
        xticklabels = df.columns
        if 'Positive' in labels[i]:
            cmap = 'Reds'
            coefs = model['coefs'][index]['pos']
        else:
            cmap = 'Blues'
            coefs = model['coefs'][index]['neg']
        data = np.full((max_features, len(df.columns)), float(coefs['other']))
        for feature, val in coefs.items():
            if feature != 'other':
                data[int(feature[2:])-1][list(df.columns).index(feature[:2])] = float(val)
    else:
        title = ''
        titlesize = 'large'
        xticklabels = [loc]
        data = np.zeros((max_features, 1))
        terms = np.zeros((len(gt_terms), 1))
        for feature, val in model['coefs'][index].items():
            if feature[:4] != 'term':
                if feature[:2] == loc:
                    data[int(feature[2:])][0] = val
            else:
                if feature.split('_')[1] == loc:
                    print(feature)
                    terms[gt_terms.index(feature.split('_')[-1])][0] = val
    sns.heatmap(data, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax, cbar=cbar)
    if mode == 'nowcasting':
        sns.heatmap(terms, ax=gt_ax, cmap=cmap, vmin=vmin, vmax=vmax, cbar=False)
    if mode == 'forecasting':
        ax.set_title(title, size=titlesize)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel('Location')
    ax.set_ylabel('Lag')
    if mode == 'nowcasting':
        gt_ax.set_title(title, size=titlesize)
        gt_ax.set_xticklabels(xticklabels)
        yticklabels = [label if i % 1 == 0 else '' for i, label in enumerate(gt_terms)]
        gt_ax.set_yticklabels(yticklabels, rotation=0)

if mode == 'forecasting':
    letter_pos = [(0.01, 0.985), (0.81, 0.985), (0.01, 0.79), (0.01, 0.59), (0.01, 0.37)]
    plt.figtext(0.91, 0.985, 'Regression Coefficients in ' + labels[0], ha='center', va='center', size='large')
else:
    letter_pos = [(0.01, 0.985), (0.68, 0.985), (0.01, 0.77), (0.01, 0.38)]
    plt.figtext(0.88, 0.985, 'Regression Coefficients in ' + labels[0], ha='center', va='center', size='large')
    ar_gt_ax.set_xlabel('Location')
plt.figtext(letter_pos[0][0], letter_pos[0][1], 'a', ha='center', va='center', size='x-large', weight='bold')
plt.figtext(letter_pos[1][0], letter_pos[1][1], 'b', ha='center', va='center', size='x-large', weight='bold')
plt.figtext(letter_pos[2][0], letter_pos[2][1], 'c', ha='center', va='center', size='x-large', weight='bold')
plt.figtext(letter_pos[3][0], letter_pos[3][1], 'd', ha='center', va='center', size='x-large', weight='bold')
plt.figtext(letter_pos[4][0], letter_pos[4][1], 'e', ha='center', va='center', size='x-large', weight='bold')


plt.tight_layout()

if mode == 'nowcasting':
    pos1 = line_plot_ax.get_position()
    line_plot_ax.set_position([pos1.x0-0.07, pos1.y0,  pos1.width+0.16, pos1.height])
    pos1 = ar_gt_ax.get_position()
    ar_gt_ax.set_position([pos1.x0+0.09, pos1.y0,  pos1.width, pos1.height])
    for ax in [armulti_gt_ax, rf_gt_ax]:
        pos1 = ax.get_position()
        ax.set_position([pos1.x0, pos1.y0-0.02,  pos1.width, pos1.height])
else:
    gru_pos_ax.set_xlabel('')
    pos1 = gru_pos_ax.get_position()
    gru_pos_ax.set_position([pos1.x0, pos1.y0-0.02,  pos1.width, pos1.height])
    plt.figtext(0.5, 0.383, 'Saliency Map for GRU for ' + runs['forecasting_gru']['dates'][index].strftime("%b. %d, %Y"), ha='center', va='center', size='large')

plt.savefig(fname)


