import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as plticker

from evaluate_models import load_data_for_evaluation, load_data2, plot_violins, plot_violins_compare, plot_lines_single, palette

# Input
geogran = 'state'

# Plotting parameters
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20
years = mdates.YearLocator()
years_format = mdates.DateFormatter('%Y')
models = ['persistance', 'forecasting_ar', 'nowcasting_ar', 'forecasting_armulti', 'nowcasting_armulti', 'forecasting_rf', 'nowcasting_rf', 'forecasting_lstm', 'nowcasting_lstm']
model_labels = ['Persistence', 'AR', 'ARGO', 'AR-net LR', 'ARGO-net LR', 'AR-net RF', 'ARGO-net RF', 'AR-net GRU', 'ARGO-net GRU']
ths = [1, 2, 4, 8]

if geogran == 'state':
	df, results = load_data_for_evaluation('state')
	loc = 'MI'
	loc_label = 'Michigan'
else:
	df, results = load_data_for_evaluation('city')
	loc = 'city111'
	loc_label = 'Fresno, CA'

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
ax9=fig.add_subplot(gs[4,0]) 
ax10=fig.add_subplot(gs[4,1]) 
ax11=fig.add_subplot(gs[4,2]) 
ax12=fig.add_subplot(gs[4,3]) 

axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12]

for ax in axes:
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)

for ax in [ax2, ax4, ax6, ax8, ax10, ax11, ax12]:
	ax.get_yaxis().set_visible(False)
	#ax.spines['left'].set_visible(False)

plt.tight_layout(rect=[0, 0.03, 1, 0.96])

plt.figtext(0.25, 0.97, 'Predictions of ILI in ' + loc_label + '\nUsing Only Historical Epi Data', ha='center', va='center', size='large')
plt.figtext(0.75, 0.97, 'Predictions of ILI in ' + loc_label + '\nUsing Historical Epi Data & GT Data', ha='center', va='center', size='large')
if geogran == 'state':
	plt.figtext(0.5, 0.23, 'Distribution of RMSE Across States', ha='center', va='center', size='large')
else:
	plt.figtext(0.5, 0.23, 'Distribution of RMSE Across Cities', ha='center', va='center', size='large')

plt.figtext(0.5, 0.945, '1 Week Reporting Delay', ha='center', va='center', size='medium')
plt.figtext(0.5, 0.768, '2 Week Reporting Delay', ha='center', va='center', size='medium')
plt.figtext(0.5, 0.59, '4 Week Reporting Delay', ha='center', va='center', size='medium')
plt.figtext(0.5, 0.415, '8 Week Reporting Delay', ha='center', va='center', size='medium')
#plt.figtext(0.5, 0.48, 'Distribution of RMSE Across Locations', ha='center', va='center', size='x-large')

plt.figtext(0.01, 0.98, 'a', ha='center', va='center', size='x-large', weight='bold')
plt.figtext(0.01, 0.22, 'b', ha='center', va='center', size='x-large', weight='bold')

for th, axes in [(1, (ax1, ax2)), (2, (ax3, ax4)), (4, (ax5, ax6)), (8, (ax7, ax8))]:
	plot_lines_single(axes[0], [results[model][th] for model in ['forecasting_ar', 'forecasting_armulti', 'forecasting_rf', 'forecasting_lstm']], loc, '',  ['AR', 'LR', 'RF', 'GRU'], ['orange', 'mediumseagreen', 'royalblue', 'orchid'])
	plot_lines_single(axes[1], [results[model][th] for model in ['nowcasting_ar', 'nowcasting_armulti', 'nowcasting_rf', 'nowcasting_lstm']], loc, '',  ['AR', 'LR', 'RF', 'GRU'], ['orange', 'mediumseagreen', 'royalblue', 'orchid'])

labels = ['AR', 'LR', 'RF', 'GRU']
for (ax, th) in [(ax9, 1), (ax10, 2), (ax11, 4), (ax12, 8)]:
	ax.set_title(str(th) + ' Week Reporting Delay', size='medium')
	plot_violins_compare(ax, [results[model][th] for model in models if 'forecasting' in model], [results[model][th] for model in models if 'nowcasting' in model], 'rmse', labels, None)

for ax in [ax10, ax11, ax12]:
	ax.get_legend().remove()

ax9.legend(labels=['Epi Data Only', 'Epi Data & GT Data'], loc='upper left')

loc = plticker.MultipleLocator(base=1.0) # this locator puts ticks at regular intervals
ax9.yaxis.set_major_locator(loc)

loc = plticker.MultipleLocator(base=2.0) # this locator puts ticks at regular intervals
for ax in [ax1, ax3, ax5, ax7]:
	ax.yaxis.set_major_locator(loc)

ax1.set_ylabel('ILI Incidence')
ax3.set_ylabel('ILI Incidence')
ax5.set_ylabel('ILI Incidence')
ax7.set_ylabel('ILI Incidence')
ax9.set_ylabel('RMSE')

for ax in [ax9, ax10, ax11, ax12]:
	pos1 = ax.get_position() 
	pos2 = [pos1.x0, pos1.y0-0.03,  pos1.width, pos1.height] 
	ax.set_position(pos2) 

ax1.legend(loc='upper left',  prop={'size': 12})

plt.savefig('figure1_state.png')
