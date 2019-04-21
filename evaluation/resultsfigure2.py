import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as plticker

from evaluate_models import load_data_for_evaluation, plot_violins, plot_violins_compare, plot_lines_single, palette

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

fig=plt.figure(figsize=(20, 11))

gs=GridSpec(4, 4, height_ratios=[1, 1, 1, 1])

ax1=fig.add_subplot(gs[0,:2]) 
ax2=fig.add_subplot(gs[0,2:]) 
ax3=fig.add_subplot(gs[1,:2]) 
ax4=fig.add_subplot(gs[1,2:]) 
ax5=fig.add_subplot(gs[2,0]) 
ax6=fig.add_subplot(gs[2,1]) 
ax7=fig.add_subplot(gs[2,2]) 
ax8=fig.add_subplot(gs[2,3]) 
ax9=fig.add_subplot(gs[3,0]) 
ax10=fig.add_subplot(gs[3,1]) 
ax11=fig.add_subplot(gs[3,2]) 
ax12=fig.add_subplot(gs[3,3]) 

axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12]

for ax in axes:
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)

for ax in [ax6, ax7, ax8, ax10, ax11, ax12]:
	ax.get_yaxis().set_visible(False)
	#ax.spines['left'].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, .98])

#plt.figtext(0.5, 0.97, 'Time-Series of 8-Week Time Horizon Predictions by Model', ha='center', va='center', size='x-large')
#plt.figtext(0.5, 0.48, 'Distribution of RMSE Across Locations', ha='center', va='center', size='x-large')

plt.figtext(0.01, 0.98, 'a', ha='center', va='center', size='x-large', weight='bold')
plt.figtext(0.01, 0.44, 'b', ha='center', va='center', size='x-large', weight='bold')

for ax in [ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12]:
	pos1 = ax.get_position() 
	pos2 = [pos1.x0, pos1.y0,  pos1.width, pos1.height-0.03] 
	ax.set_position(pos2) 

for ax in [ax5, ax6, ax7, ax8]:
	pos1 = ax.get_position() 
	pos2 = [pos1.x0, pos1.y0-0.03,  pos1.width, pos1.height] 
	ax.set_position(pos2) 

plot_lines_single(ax1, [results_states[model][8] for model in ['forecasting_ar', 'forecasting_armulti', 'forecasting_rf', 'forecasting_lstm']], 'WA', 'Forecasts (Historical Epi Data Only) for ILI in Washington State',  ['AR', 'AR-net LR', 'AR-net RF', 'AR-net GRU'], ['orange', 'mediumseagreen', 'royalblue', 'orchid'])
plot_lines_single(ax2, [results_states[model][8] for model in ['nowcasting_ar', 'nowcasting_armulti', 'nowcasting_rf', 'nowcasting_lstm']], 'WA', 'Nowcasts (Historical Epi Data & Real-Time GT Data) for ILI in Washington State',  ['ARGO', 'ARGO-net LR', 'ARGO-net RF', 'ARGO-net GRU'], ['orange', 'mediumseagreen', 'royalblue', 'orchid'])
plot_lines_single(ax3, [results_cities[model][8] for model in ['forecasting_ar', 'forecasting_armulti', 'forecasting_rf', 'forecasting_lstm']], 'city135', 'Forecasts (Historical Epi Data Only) for ILI in Huntsville, AL',  ['AR', 'AR-net LR', 'AR-net RF', 'AR-net GRU'], ['orange', 'mediumseagreen', 'royalblue', 'orchid'])
plot_lines_single(ax4, [results_cities[model][8] for model in ['nowcasting_ar', 'nowcasting_armulti', 'nowcasting_rf', 'nowcasting_lstm']], 'city135', 'Nowcasts (Historical Epi Data & Real-Time GT Data) for ILI in Huntsville, AL',  ['ARGO', 'ARGo-net LR', 'ARGO-net RF', 'ARGO-net GRU'], ['orange', 'mediumseagreen', 'royalblue', 'orchid'])

labels = ['AR(GO)', 'AR(GO)-net LR', 'AR(GO)-net RF', 'AR(GO)-net GRU']
for (ax, th) in [(ax5, 1), (ax6, 2), (ax7, 4), (ax8, 8)]:
	plot_violins_compare(ax, [results_states[model][th] for model in models if 'forecasting' in model], [results_states[model][th] for model in models if 'nowcasting' in model], 'rmse', labels, None)
	ax.set_title(str(th) + ' Week Time Horizon')
for (ax, th) in [(ax9, 1), (ax10, 2), (ax11, 4), (ax12, 8)]:
	plot_violins_compare(ax, [results_cities[model][th] for model in models if 'forecasting' in model], [results_cities[model][th] for model in models if 'nowcasting' in model], 'rmse', labels, None)

for ax in [ax6, ax7, ax8, ax9, ax10, ax11, ax12]:
	ax.get_legend().remove()

ax5.legend(labels=['No GT Data', 'Including GT Data'], loc='upper left')

for ax in [ax9, ax10, ax11, ax12]:
	ax.tick_params(axis='x', which='major', labelsize=10)
	ax.tick_params(axis='x', which='minor', labelsize=10)

for ax in [ax5, ax9]:
	loc = plticker.MultipleLocator(base=1.0) # this locator puts ticks at regular intervals
	ax.yaxis.set_major_locator(loc)

for ax in [ax5, ax6, ax7, ax8]:
	ax.set_xticklabels([])

loc = plticker.MultipleLocator(base=1.0) # this locator puts ticks at regular intervals
ax5.yaxis.set_major_locator(loc)
ax9.yaxis.set_major_locator(loc)

ax1.set_ylabel('ILI Incidence')
ax3.set_ylabel('ILI Incidence')
ax5.set_ylabel('State-Level RMSE')
ax9.set_ylabel('City-Level RMSE')

ax1.legend(loc='upper left',  prop={'size': 12})
ax2.legend(loc='upper left',  prop={'size': 12})

plt.savefig('bla.png')