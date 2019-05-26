import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as plticker

from evaluate_models import load_data_for_evaluation, plot_violins, plot_violins_compare, plot_lines_single, palette

# Input
geogran = 'state'

# Plotting parameters
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16
years = mdates.YearLocator()
years_format = mdates.DateFormatter('%Y')
models = ['persistance', 'forecasting_ar', 'nowcasting_ar', 'forecasting_armulti', 'nowcasting_armulti', 'forecasting_rf', 'nowcasting_rf', 'forecasting_lstm', 'nowcasting_lstm']
model_labels = ['Persistence', 'AR', 'ARGO', 'AR-net LR', 'ARGO-net LR', 'AR-net RF', 'ARGO-net RF', 'AR-net GRU', 'ARGO-net GRU']
ths = [1, 2, 4, 8]

if geogran == 'state':
	df, results = load_data_for_evaluation('state')
	loc = 'NY'
	loc_label = 'New York State'
else:
	df, results = load_data_for_evaluation('city')
	loc = 'city2'
	loc_label = 'Abilene, TX'

fig=plt.figure(figsize=(20, 14))

gs=GridSpec(5, 4, height_ratios=[6, 6, 6, 6, 7])

ax1=fig.add_subplot(gs[0,:2]) 
ax2=fig.add_subplot(gs[0,2:]) 
ax3=fig.add_subplot(gs[1,:2]) 
ax4=fig.add_subplot(gs[1,2:]) 
ax5=fig.add_subplot(gs[2,:2]) 
ax6=fig.add_subplot(gs[2,2:]) 
ax7=fig.add_subplot(gs[3,:2]) 
ax8=fig.add_subplot(gs[3,2:]) 
ax9=fig.add_subplot(gs[4,:2]) 
ax10=fig.add_subplot(gs[4,2:]) 


axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10]

for ax in axes:
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)

for ax in [ax2, ax4, ax6, ax8]:
	ax.get_yaxis().set_visible(False)
	#ax.spines['left'].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 1])

plt.figtext(0.01, 0.99, 'a', ha='center', va='center', size='x-large', weight = 'bold')
plt.figtext(0.01, 0.2, 'b', ha='center', va='center', size='x-large', weight = 'bold')

for th, axes in [(1, (ax1, ax2)), (2, (ax3, ax4)), (4, (ax5, ax6)), (8, (ax7, ax8))]:
	plot_lines_single(axes[0], [results[model][th] for model in ['forecasting_ar', 'forecasting_armulti']], loc, str(th) + '-Week Forecasts (Epi Data Only) in ' + loc_label,  ['AR', 'AR-net LR'], ['orange', 'mediumseagreen', 'royalblue', 'orchid'])
	plot_lines_single(axes[1], [results[model][th] for model in ['nowcasting_ar', 'nowcasting_armulti']], loc, str(th) + '-Week Nowcasts (Epi Data & GT Data) in ' + loc_label,  ['ARGO', 'ARGO-net LR'], ['orange', 'mediumseagreen', 'royalblue', 'orchid'])
plot_violins_compare(ax9, [results['forecasting_ar'][th] for th in [1, 2, 4, 8]], [results['forecasting_armulti'][th] for th in [1, 2, 4, 8]], 'rmse', ['1', '2', '4', '8'], ['orange', 'mediumseagreen'])
plot_violins_compare(ax10, [results['nowcasting_ar'][th] for th in [1, 2, 4, 8]], [results['nowcasting_armulti'][th] for th in [1, 2, 4, 8]], 'rmse', ['1', '2', '4', '8'], ['orange', 'mediumseagreen'])

ax9.legend(labels=['AR', 'LR'], loc='upper left')
ax10.legend(labels=['AR', 'LR'], loc='upper left')

loc = plticker.MultipleLocator(base=1.0) # this locator puts ticks at regular intervals
ax9.yaxis.set_major_locator(loc)
loc = plticker.MultipleLocator(base=3.0) # this locator puts ticks at regular intervals
for ax in [ax1, ax3, ax5, ax7]:
	ax.yaxis.set_major_locator(loc)


ax1.set_ylabel('ILI Incidence')
ax3.set_ylabel('ILI Incidence')
ax5.set_ylabel('ILI Incidence')
ax7.set_ylabel('ILI Incidence')
ax9.set_ylabel('RMSE')

ax9.set_title('Forecasts')
ax10.set_title('Nowcasts')

for ax in [ax9, ax10]:
	ax.set_xlabel('Time Horizon of Prediction (weeks)')

ax1.legend(loc='upper left')
ax2.legend(loc='upper left')

plt.savefig('figure2_' + geogran + '.png')
