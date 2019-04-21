import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as plticker

from evaluate_models import load_data_for_evaluation, plot_violins_compare

# Plotting parameters
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
years = mdates.YearLocator()
years_format = mdates.DateFormatter('%Y')
models = ['persistance', 'forecasting_ar', 'nowcasting_ar', 'forecasting_armulti', 'nowcasting_armulti', 'forecasting_rf', 'nowcasting_rf', 'forecasting_lstm', 'nowcasting_lstm']
model_labels = ['Persistence', 'AR', 'ARGO', 'AR-net LR', 'ARGO-net LR', 'AR-net RF', 'ARGO-net RF', 'AR-net GRU', 'ARGO-net GRU']
ths = [1, 2, 4, 8]

df_states, results_states = load_data_for_evaluation('state')
df_cities, results_cities = load_data_for_evaluation('city')

labels = ['AR(GO)', 'AR(GO)-net LR', 'AR(GO)-net RF', 'AR(GO)-net GRU']
fig, ax = plt.subplots(4, 2, figsize=(20, 15))
for t, th in enumerate(ths):
	plot_violins_compare(ax[t, 0], [results_states[model][th] for model in models if 'forecasting' in model], [results_states[model][th] for model in models if 'nowcasting' in model], 'rmse', labels, None, 'state')
	plot_violins_compare(ax[t, 1], [results_cities[model][th] for model in models if 'forecasting' in model], [results_cities[model][th] for model in models if 'nowcasting' in model], 'rmse', labels, None, 'city')
	ax[t, 0].get_legend().remove()
	if t != 0:
		ax[t, 1].get_legend().remove()
		ax[0, 0].set_title(str(th) + ' Week Time Horizon')
		ax[0, 1].set_title(str(th) + ' Week Time Horizon')
ax[0, 0].set_title('State-Level Dataset \n 1 Week Time Horizon')
ax[0, 1].set_title('City-Level Dataset \n 1 Week Time Horizon')
ax[0, 0].set_ylabel('RMSE')
ax[0, 1].legend(loc='upper right')
L=ax[0, 1].legend()
L.get_texts()[0].set_text('Forecasting (no GT data)')
L.get_texts()[1].set_text('Nowcasting (with GT data)')
plt.tight_layout(rect=[0, 0.01, 1, 0.93])
plt.suptitle('Comparing RMSE for Forecasting (Epi Data Only) and Nowcasting (Including GT Data)', y = 0.97, fontsize='x-large')
plt.savefig('bla.png')