import sys
import json
import operator
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as plticker

sys.path.insert(0, '../')
from utils import load_flu, load_geo, load_run, crop_run, load_dengue, load_flu_states, load_flu_cities_subset

# Plotting parameters
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
years = mdates.YearLocator()
years_format = mdates.DateFormatter('%Y')
models = ['persistance', 'forecasting_ar', 'nowcasting_ar', 'forecasting_armulti', 'nowcasting_armulti', 'forecasting_rf', 'nowcasting_rf', 'forecasting_lstm', 'nowcasting_lstm']
model_labels = ['Persistence', 'AR', 'ARGO', 'AR-net LR', 'ARGO-net LR', 'AR-net RF', 'ARGO-net RF', 'AR-net GRU', 'ARGO-net GRU']
ths = [1, 2, 4, 8]

palette = {}
for l in model_labels:
	if 'GRU' in l:
		palette[l] = 'orchid'
	elif 'RF' in l:
		palette[l] = 'royalblue'
	elif 'LR' in l:
		palette[l] = 'mediumseagreen'
	elif 'AR' in l:
		palette[l] = 'orange'
	else:
		palette[l] = 'indianred'

palette['AR'] = 'orange'
palette['LR'] = 'mediumseagreen'
palette['RF'] = 'royalblue'
palette['GRU'] = 'mediumpurple'
palette['Pers.'] = 'indianred'

palette_flat = ['indianred', 'orange', 'mediumseagreen', 'royalblue', 'orchid']

# Load original city- and state-level data

def load_data_for_evaluation(geogran):
	# Load original data from city- and state-level
	df_cities = load_flu_cities_subset('../')
	df_states = load_flu_states('../')
	num_eval_cities = int((len(df_cities) - 52)*(5/10))
	num_eval_states = int((len(df_states) - 52)*(5/10))
	# Load model runs
	results = {model:{} for model in models}
	for model in models:
		for th in ths:
			if geogran == 'state':
				results[model][th] = load_run('../results/no_coefs/results_states/' + model + '/' + str(th) + '.json', num_eval_states)
			else:
				results[model][th] = load_run('../results/no_coefs/results_cities/' + model + '/' + str(th) + '.json', num_eval_cities)
	if geogran == 'state':
		return df_states, results
	else:
		return df_cities, results

def load_data2(geogran):
	# Load original data from city- and state-level
	df_cities = load_flu_cities_subset('../')
	df_states = load_flu_states('../')
	num_eval_cities = int((len(df_cities) - 52)*(2/4))
	num_eval_states = int((len(df_states) - 52)*(2/4))
	# Load model runs
	results = {model:{} for model in models}
	for model in models:
		for th in ths:
			if geogran == 'state':
				if model == 'forecasting_lstm' or model == 'nowcasting_lstm':
					results[model][th] = load_run('../new_new_results_state/' + str(th) + '/' + model + '.json', num_eval_states)
				else:
					results[model][th] = load_run('../results/no_coefs/results_states/' + model + '/' + str(th) + '.json', num_eval_states)
			else:
				if model == 'forecasting_lstm':
					results[model][th] = load_run('../new_results_city/' + str(th) + '/' + model + '.json', num_eval_cities)
				elif model == 'nowcasting_lstm':
					results[model][th] = load_run('../new_new_results_city/' + str(th) + '/' + model + '.json', num_eval_cities)
				else:
					results[model][th] = load_run('../results/no_coefs/results_cities/' + model + '/' + str(th) + '.json', num_eval_cities)
	if geogran == 'state':
		return df_states, results
	else:
		return df_cities, results

# Violin Plot
def plot_violins(ax, models, metric, labels, colors=None):
    if metric == 'corr':
        model_evals = [[np.corrcoef(preds[city]['ytrues'], preds[city]['yhats'])[0][1] for city in preds.keys()] for preds in models]
    else:
        model_evals = [[np.sqrt(mean_squared_error(preds[city]['ytrues'], preds[city]['yhats'])) for city in preds.keys()] for preds in models]
    results = pd.DataFrame(model_evals).transpose()
    results.columns = labels
    if colors is None:
    	g = sns.violinplot(data=results, ax=ax)
    else:
        g = sns.violinplot(data=results, ax=ax, palette=colors)
        for e, model_eval in enumerate(model_evals):
            ax.axhline(np.median(model_eval), color=palette_flat[e], xmax=0.02, linewidth=2)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if metric == 'rmse':
    	ax.set_ylim(0, 2)
    else:
    	ax.set_ylim(0, 1)

def plot_violins_compare(ax, forecasting_models, nowcasting_models, metric, labels, colors=None, geogran='state'):
    if metric == 'corr':
        model_evals_forecasting = [[np.corrcoef(preds[city]['ytrues'], preds[city]['yhats'])[0][1] for city in preds.keys()] for preds in forecasting_models]
        model_evals_nowcasting = [[np.corrcoef(preds[city]['ytrues'], preds[city]['yhats'])[0][1] for city in preds.keys()] for preds in nowcasting_models]
    else:
        model_evals_forecasting = [[np.sqrt(mean_squared_error(preds[city]['ytrues'], preds[city]['yhats'])) for city in preds.keys()] for preds in forecasting_models]
        model_evals_nowcasting = [[np.sqrt(mean_squared_error(preds[city]['ytrues'], preds[city]['yhats'])) for city in preds.keys()] for preds in nowcasting_models]
    results1 = pd.DataFrame(model_evals_forecasting).transpose()
    results1.columns = labels
    results1 = results1.melt(var_name='groups', value_name='vals')
    results1['Digital Data?'] = [False for _ in range(len(results1))]
    results2 = pd.DataFrame(model_evals_nowcasting).transpose()
    results2.columns = labels
    results2 = results2.melt(var_name='groups', value_name='vals')
    results2['Digital Data?'] = [True for _ in range(len(results2))]
    results = pd.concat([results1, results2], ignore_index=True)
    if colors is None:
    	g = sns.violinplot(data=results, x='groups', y='vals', hue='Digital Data?', ax=ax, split=True, inner=None, palette="Pastel1")
    else:
    	g = sns.violinplot(data=results, x='groups', y='vals', hue='Digital Data?', ax=ax, split=True, inner=None, palette=colors)
    ax.set_ylim(0, 2)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    #for tick in ax.get_xticklabels():
    #    tick.set_fontsize('medium') 
    #    tick.set_rotation(45)
    #if geogran == 'state':
    #	ax.set_title('State-Level Dataset')
    #else:
    #	ax.set_title('City-Level Dataset')

# Line plot
def plot_lines(ax, models, cities, titles, labels, colors):
	corrs = {}
	for i, model in enumerate(models):
		for c, city in enumerate(cities):
			dates = [pd.to_datetime(date) for date in models[i][city]['dates']]
			if i == 0:
				ax[c].fill_between(dates, models[i][city]['ytrues'], color='lightgrey', label='Ground truth')
			if colors[i] == 'black':
				ax[c].plot(dates, models[i][city]['yhats'], label=labels[i], color=colors[i], linewidth=3, dashes=[2, 2])
			else:
				ax[c].plot(dates, models[i][city]['yhats'], label=labels[i], color=colors[i], linewidth=3)
			ax[c].set_title(titles[c])
			ax[c].xaxis.set_major_locator(years)
			ax[c].xaxis.set_major_formatter(years_format)
			ax[c].spines['right'].set_visible(False)
			ax[c].spines['top'].set_visible(False)
			if i == len(models) - 1:
				corrs[city] = np.corrcoef(models[i][city]['ytrues'], models[i][city]['yhats'])[0][1]
	sorted_corrs = sorted(corrs.items(), key=operator.itemgetter(1))
	print(len(sorted_corrs))
	for i, (city, corr) in enumerate(sorted_corrs):
		if i % 10 == 0:
			print(city + ': %.2f' % corr)
	ax[0].legend(loc='best')

def plot_lines_single(ax, models, city, title, labels, colors):
	for i, model in enumerate(models):
		dates = [pd.to_datetime(date) for date in models[i][city]['dates']]
		if i == 0:
			ax.fill_between(dates, models[i][city]['ytrues'], color='lightgrey', label='Ground truth')
		ax.plot(dates, models[i][city]['yhats'], label=labels[i], color=colors[i], linewidth=3)
		ax.set_xlim(min(dates), max(dates))
	ax.set_title(title, fontsize='medium')
	ax.xaxis.set_major_locator(years)
	ax.xaxis.set_major_formatter(years_format)
	#ax.legend(loc='best')
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.set_ylim(0)

def plot_evolution(ax, models_by_th, city, colors, title):
	for t, th in enumerate(models_by_th.keys()):
		model = models_by_th[th]
		dates = [pd.to_datetime(date) for date in model[city]['dates']]
		if th == 1:
			ax.fill_between(dates, model[city]['ytrues'], color='lightgrey', label='Ground truth')
		ax.plot(dates, model[city]['yhats'], color=colors[t], label=str(th) + ' week', linewidth=3)
		ax.set_xlim(min(dates), max(dates))
		#ax.set_ylim(0, max([max([max(model[city]['ytrues']), max(model[city]['yhats'])]) for model in models_by_th.values()]))
	ax.set_title(title, fontsize='medium')
	ax.xaxis.set_major_locator(years)
	ax.xaxis.set_major_formatter(years_format)
	ax.set_ylim(0)
	#ax.legend(loc='best')
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)

evolution_palette = ['mediumblue', 'forestgreen', 'gold', 'mediumvioletred']
