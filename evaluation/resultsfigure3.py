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

fig=plt.figure(figsize=(20, 9))

gs=GridSpec(3, 4, height_ratios=[3, 3, 4])

ax1=fig.add_subplot(gs[0,:2]) 
ax2=fig.add_subplot(gs[0,2:]) 
ax3=fig.add_subplot(gs[1,:2]) 
ax4=fig.add_subplot(gs[1,2:]) 
ax5=fig.add_subplot(gs[2,0]) 
ax6=fig.add_subplot(gs[2,1]) 
ax7=fig.add_subplot(gs[2,2]) 
ax8=fig.add_subplot(gs[2,3]) 

axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]

for ax in axes:
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)

for ax in [ax6, ax7, ax8]:
	ax.get_yaxis().set_visible(False)
	#ax.spines['left'].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.98])

plt.figtext(0.01, 0.98, 'a', ha='center', va='center', size='x-large', weight = 'bold')
plt.figtext(0.01, 0.33, 'b', ha='center', va='center', size='x-large', weight = 'bold')

for ax in [ax5, ax6, ax7, ax8]:
	pos1 = ax.get_position() 
	pos2 = [pos1.x0, pos1.y0+0.02,  pos1.width, pos1.height-0.08] 
	ax.set_position(pos2) 

plot_lines_single(ax1, [results_states[model][8] for model in ['forecasting_ar', 'forecasting_armulti']], 'NY', 'Forecasts (Historical Epi Data Only) in New York State',  ['AR(GO)', 'AR(GO)-net LR'], ['orange', 'mediumseagreen', 'royalblue', 'orchid'])
plot_lines_single(ax2, [results_states[model][8] for model in ['nowcasting_ar', 'nowcasting_armulti']], 'SD', 'Nowcasts (Historical Epi Data & Real-Time GT Data) in South Dakota',  ['AR(GO)', 'AR(GO)-net LR'], ['orange', 'mediumseagreen', 'royalblue', 'orchid'])
plot_lines_single(ax3, [results_cities[model][8] for model in ['forecasting_ar', 'forecasting_armulti']], 'city201', 'Forecasts (Historical Epi Data Only) in Nashville, TN',  ['AR(GO)', 'AR(GO)-net LR'], ['orange', 'mediumseagreen', 'royalblue', 'orchid'])
plot_lines_single(ax4, [results_cities[model][8] for model in ['nowcasting_ar', 'nowcasting_armulti']], 'city266', 'Nowcasts (Historical Epi Data & Real-Time GT Data) in Santa Barbara, CA',  ['AR(GO)', 'AR(GO)-net LR'], ['orange', 'mediumseagreen', 'royalblue', 'orchid'])

plot_violins_compare(ax5, [results_states['forecasting_ar'][th] for th in [1, 2, 4, 8]], [results_states['forecasting_armulti'][th] for th in [1, 2, 4, 8]], 'rmse', ['1', '2', '4', '8'], ['orange', 'mediumseagreen'])
plot_violins_compare(ax6, [results_cities['forecasting_ar'][th] for th in [1, 2, 4, 8]], [results_cities['forecasting_armulti'][th] for th in [1, 2, 4, 8]], 'rmse', ['1', '2', '4', '8'], ['orange', 'mediumseagreen'])
plot_violins_compare(ax7, [results_states['nowcasting_ar'][th] for th in [1, 2, 4, 8]], [results_states['nowcasting_armulti'][th] for th in [1, 2, 4, 8]], 'rmse', ['1', '2', '4', '8'], ['orange', 'mediumseagreen'])
plot_violins_compare(ax8, [results_cities['nowcasting_ar'][th] for th in [1, 2, 4, 8]], [results_cities['nowcasting_armulti'][th] for th in [1, 2, 4, 8]], 'rmse', ['1', '2', '4', '8'], ['orange', 'mediumseagreen'])

ax5.legend(labels=['AR(GO)', 'AR(GO)-net LR'], loc='upper left')

loc = plticker.MultipleLocator(base=1.0) # this locator puts ticks at regular intervals
ax5.yaxis.set_major_locator(loc)

ax1.set_ylabel('ILI Incidence')
ax3.set_ylabel('ILI Incidence')
ax5.set_ylabel('RMSE')

ax5.set_title('State-Level Forecasts')
ax6.set_title('City-Level Forecasts')
ax7.set_title('State-Level Nowcasts')
ax8.set_title('City-Level Nowcasts')

for ax in [ax5, ax6, ax7, ax8]:
	ax.set_xlabel('Time Horizon of Prediction (weeks)')

ax1.legend(loc='best')

for ax in [ax6, ax7, ax8]:
	ax.get_legend().remove()

plt.savefig('bla.png')
