import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as plticker

from evaluate_models import load_data_for_evaluation, plot_violins, palette, palette_flat

# Plotting parameters
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
years = mdates.YearLocator()
years_format = mdates.DateFormatter('%Y')
models = ['persistance', 'forecasting_ar', 'nowcasting_ar', 'forecasting_armulti', 'nowcasting_armulti', 'forecasting_rf', 'nowcasting_rf', 'forecasting_lstm', 'nowcasting_lstm']
model_labels = ['Persistence', 'AR', 'ARGO', 'AR-net LR', 'ARGO-net LR', 'AR-net RF', 'ARGO-net RF', 'AR-net GRU', 'ARGO-net GRU']
ths = [1, 2, 4, 8]

df_states, results_states = load_data_for_evaluation('state')

fig, ax = plt.subplots(2, 2, figsize=(20, 10))
ax = ax.flatten()
for i in range(len(ths)):
	th = ths[i]
	plot_violins(ax[i], [results_states[model][th] for model in models if model == 'persistance' or 'forecasting' in model], 'corr', [label for label in model_labels if label == 'Persistence' or 'ARGO' in label], palette)
	ax[i].set_title('Time horizon: ' + str(th) + ' weeks')
	ax[i].set_ylim(0, 1)
	for tick in ax[i].get_xticklabels():
		tick.set_fontsize('small') 
ax[0].set_ylabel('corr')
plt.tight_layout(rect=[0, 0.01, 1, 0.93])
plt.suptitle('Distribution of correlations across cities by model & time horizon ', y = 0.97, fontsize='x-large')
plt.savefig('bla.png')

