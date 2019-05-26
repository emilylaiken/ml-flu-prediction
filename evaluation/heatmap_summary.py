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

geogran = 'state'
th1 = 1 # 1, 2, 4, or 8
th2 = 8
loc = 'PA' # Name of city/state
df = load_flu_states('../')
models = ['forecasting_ar', 'forecasting_gru']
lineplotlabels = ['AR', 'GRU']
colors = ['orange','mediumpurple']
max_features = 52
num_eval = int((len(df) - 52)*(5/10))
fname = 'figure5.png'

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20
years = mdates.YearLocator()
years_format = mdates.DateFormatter('%Y')

fig=plt.figure(figsize=(20, 23))
gs=GridSpec(7, 8, height_ratios=[5, 4, 4, 4, 4, 4, 4])
line_plot_ax=fig.add_subplot(gs[0,:6]) 
ar_ax1=fig.add_subplot(gs[0,6:7]) 
ar_ax2=fig.add_subplot(gs[0,7:]) 
armulti_ax1=fig.add_subplot(gs[1,:]) 
armulti_ax2=fig.add_subplot(gs[2,:]) 
rf_ax1=fig.add_subplot(gs[3,:])  
rf_ax2=fig.add_subplot(gs[4,:])
gru_ax1=fig.add_subplot(gs[5,:]) 
gru_ax2=fig.add_subplot(gs[6,:])    

for th in [th1, th2]:
    runs = {}
    for model in models:
        runs[model] = {}
        run = load_run('../results/' + geogran + '/' + str(th) + '/' + loc + '/' + model + '.json')
        runs[model]['dates'] = [pd.to_datetime(date) for date in run['dates']][-num_eval:]
        runs[model]['ytrues'] = run['ytrues'][-num_eval:]
        runs[model]['yhats'] = run['yhats'][-num_eval:]
        runs[model]['coefs'] = run['coefs'][-num_eval:]
    for i, model in enumerate(models):
        if i == 0 and th == th1:
            line_plot_ax.fill_between(runs[model]['dates'], runs[model]['ytrues'], color='lightgrey', label='Ground truth')
            line_plot_ax.set_ylim(0, max(runs[model]['ytrues']))
        if th == th1:
            line_plot_ax.plot(runs[model]['dates'], runs[model]['yhats'], label=lineplotlabels[i] + ', RD = ' + str(th), color=colors[i], linewidth=3)
        else:
            line_plot_ax.plot(runs[model]['dates'], runs[model]['yhats'], label=lineplotlabels[i] + ', RD = ' + str(th), dashes=[2, 2], color=colors[i], linewidth=3)
    line_plot_ax.set_xlim(min(runs[models[0]]['dates']), max(runs[models[0]]['dates']))


line_plot_ax.xaxis.set_major_locator(years)
line_plot_ax.xaxis.set_major_formatter(years_format)
line_plot_ax.legend(loc='upper left', prop={'size': 12})
line_plot_ax.spines['right'].set_visible(False)
line_plot_ax.spines['top'].set_visible(False)
line_plot_ax.set_ylabel('ILI Incidence')

for model, ax1, ax2, tag, cbar, cmin, cmax, cmap, xlabels, title, titlesize in [('ar', ar_ax1, ar_ax2, '', True, -0.5, 0.5, 'RdBu_r', [loc], 'Average AR Coeffients', 'medium'), 
                                                                        ('lr', armulti_ax1, armulti_ax2, '', False, -0.5, 0.5, 'RdBu_r', df.columns, 'Average LR Coefficients', 'medium'), 
                                                                        ('rf', rf_ax1, rf_ax2, '', False, -0.5, 0.5, 'RdBu_r', df.columns, 'Average RF Feature Importances', 'medium'), 
                                                                        ('gru', gru_ax1, gru_ax2, '_pos', False, -1, 1, 'RdBu_r', df.columns, 'Positive Gradients', 'medium')]:
    data1 = np.load('../results/avg_heatmaps/' + geogran + '/' + str(th1) + '/' + model + '/' + loc + tag + '.npy')
    sns.heatmap(data1, ax=ax1, cmap=cmap, vmin=cmin, vmax=cmax, cbar=False)
    data2 = np.load('../results/avg_heatmaps/' + geogran + '/' + str(th2) + '/' + model + '/' + loc + tag + '.npy')
    sns.heatmap(data2, ax=ax2, cmap=cmap, vmin=cmin, vmax=cmax, cbar=cbar)
    if ax1 == ar_ax1:
        ax1.set_title(str(th1) + ' Week RD', size='medium')
        ax2.set_title(str(th2) + ' Week RD', size='medium')
    else:
        ax1.set_title(str(th1) + ' Week Reporting Delay', size='medium')
        ax2.set_title(str(th2) + ' Week Reporting Delay', size='medium')
    for ax in [ax1, ax2]:
        if ax in [ar_ax1, ar_ax2, armulti_ax2, rf_ax2, gru_ax2]:
            ax.set_xlabel('Location')
        ax.set_xticklabels(xlabels, size='small')
        ax.set_ylabel('Lag')


plt.tight_layout(rect=[0, 0, 1, 0.99])

for ax in [armulti_ax1, rf_ax1, gru_ax1]:
    pos1 = ax.get_position()
    ax.set_position([pos1.x0, pos1.y0-0.02,  pos1.width, pos1.height])


plt.figtext(0.5, 0.81, 'Average Coefficients in LR', ha='center', va='center', size='large')
plt.figtext(0.5, 0.536, 'Average Feature Importances in RF', ha='center', va='center', size='large')
plt.figtext(0.5, 0.264, 'Average Saliency Maps for GRU (Positive Gradients)', ha='center', va='center', size='large')
plt.figtext(0.4, 0.99, 'Preidctions for ILI in Virginia Assuming 1 & 8 Week Reporting Delays', ha='center', va='center', size='large')
plt.figtext(0.86, 0.99, 'Average Coefficients in AR', ha='center', va='center', size='large')

''' 
letter_pos = [(0.01, 0.985), (0.84, 0.985), (0.01, 0.79), (0.01, 0.59), (0.01, 0.37)]
plt.figtext(letter_pos[0][0], letter_pos[0][1], 'a', ha='center', va='center', size='x-large', weight='bold')
plt.figtext(letter_pos[1][0], letter_pos[1][1], 'b', ha='center', va='center', size='x-large', weight='bold')
plt.figtext(letter_pos[2][0], letter_pos[2][1], 'c', ha='center', va='center', size='x-large', weight='bold')
plt.figtext(letter_pos[3][0], letter_pos[3][1], 'd', ha='center', va='center', size='x-large', weight='bold')
plt.figtext(letter_pos[4][0], letter_pos[4][1], 'e', ha='center', va='center', size='x-large', weight='bold')
'''
plt.savefig(fname)


