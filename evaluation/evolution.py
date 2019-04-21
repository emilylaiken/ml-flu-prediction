import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as plticker

from evaluate_models import load_data_for_evaluation, plot_evolution, evolution_palette

# Plotting parameters
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
years = mdates.YearLocator()
years_format = mdates.DateFormatter('%Y')
models = ['persistance', 'forecasting_ar', 'nowcasting_ar', 'forecasting_armulti', 'nowcasting_armulti', 'forecasting_rf', 'nowcasting_rf', 'forecasting_lstm', 'nowcasting_lstm']
model_labels = ['Persistence', 'AR', 'ARGO', 'AR-net LR', 'ARGO-net LR', 'AR-net RF', 'ARGO-net RF', 'AR-net GRU', 'ARGO-net GRU']
ths = [1, 2, 4, 8]

df_cities, results = load_data_for_evaluation('city')

fig, ax = plt.subplots(4, 1, figsize=(20, 23))
i = 0
for  m, model in enumerate(models):
	if 'forecasting' in model:
		plot_evolution(ax[i], {th: results[model][th] for th in ths}, 'city120', evolution_palette, model_labels[m])
		i = i + 1
ax[0].set_ylabel('ILI Level')
ax[0].legend(loc='best')
plt.tight_layout(rect=[0, 0.01, 1, 0.95])
plt.suptitle('Evolution of Model Forecasts in Charlotte', y = 0.97, fontsize='x-large')
plt.savefig('bla.png')