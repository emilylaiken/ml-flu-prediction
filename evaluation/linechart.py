import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as plticker

from evaluate_models import load_data_for_evaluation, plot_lines, plot_lines_single
from utils import load_geo

# Plotting parameters
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
years = mdates.YearLocator()
years_format = mdates.DateFormatter('%Y')
models = ['persistance', 'forecasting_ar', 'nowcasting_ar', 'forecasting_armulti', 'nowcasting_armulti', 'forecasting_rf', 'nowcasting_rf', 'forecasting_lstm', 'nowcasting_lstm']
model_labels = ['Persistence', 'AR', 'ARGO', 'AR-net LR', 'ARGO-net LR', 'AR-net RF', 'ARGO-net RF', 'AR-net GRU', 'ARGO-net GRU']
ths = [1, 2, 4, 8]

df_cities, results = load_data_for_evaluation('city')


th_to_graph = 8
fig, ax = plt.subplots(1, figsize=(20, 5))
models_to_graph, model_labels_to_graph = [], []
for m, model in enumerate(models):
	if 'forecasting' in model:
		models_to_graph.append(model)
		model_labels_to_graph.append(model_labels[m])
plot_lines_single(ax, [results[model][th_to_graph] for model in models_to_graph], 'city105', '8 Week Time Horizon Forecasts (Historical Epidemiological Data Only) for ILI in Fort Myers, FL',  model_labels_to_graph, ['orange', 'mediumseagreen', 'royalblue', 'orchid'])
ax.set_ylabel('ILI Level')
plt.tight_layout()
plt.savefig('bla.png')


'''
locs = ['city172', 'city95', 'city80', 'city165', 'city303', 'city30', 'city300', 'city227', 'city249', 'city123', 'city308', 'city120', 'city103', 'city186', 'city174', 'city133']
#locs = ['city95', 'city51', 'city168', 'city266', 'city40', 'city143', 'city293', 'city52', 'city173', 'city135', 'city136', 'city230', 'city27', 'city108', 'city11', 'city69']
#locs = ['WV', 'TN', 'WA', 'MA', 'TX', 'NY', 'KY', 'UT', 'NC', 'WI', 'OH', 'MI', 'OR', 'GA', 'MD', 'NE']
#locs = ['KS', 'TN', 'VA', 'NM', 'TX', 'WA', 'MA', 'UT', 'NY', 'MN', 'GA', 'NH', 'MI', 'ND', 'AK', 'MD']
geo = load_geo('../')
for mode in ['nowcasting']:
	for th_to_graph in [1, 2, 4, 8]:
		models_to_graph, model_labels_to_graph = [], []
		for m, model in enumerate(models):
			if mode in model:
				models_to_graph.append(model)
				model_labels_to_graph.append(model_labels[m])
		fig, ax = plt.subplots(8, 2, figsize=(20, 25))
		ax = ax.flatten()
		titles = [geo.loc[int(city[4:])]['scf_name'] + ', ' + geo.loc[int(city[4:])]['state_acronym'] for city in locs]
		#plot_lines(ax, [results[model][th_to_graph] for model in models_to_graph], df.columns[:len(ax)], df.columns[:len(ax)], model_labels[1:], ['orange', 'mediumseagreen', 'royalblue', 'orchid'])
		plot_lines(ax, [results[model][th_to_graph] for model in models_to_graph], locs, titles, model_labels_to_graph, ['orange', 'mediumseagreen', 'royalblue', 'orchid'])
		ax[0].set_ylabel('Flu level')
		plt.tight_layout(rect=[0, 0.01, 1, 0.98])
		plt.suptitle(str(th_to_graph) + ' Week Time Horizon Nowcasts (Historical Epidemiological Data and GT Data) for ILI in ' + str(len(ax)) + ' Cities', y = 0.99, fontsize='xx-large')
		#plt.suptitle(str(th_to_graph) + ' Week Time Horizon Forecasts (Historical Epidemiological Data Only) for ILI in ' + str(len(ax)) + ' Cities', y = 0.99, fontsize='xx-large')
		plt.savefig('bla.png')
'''