import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as plticker

from evaluate_models import load_data_for_evaluation, plot_violins, plot_violins_compare, plot_evolution, evolution_palette, palette

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
	loc = 'OR'
	loc_label = 'Oregon'
else:
	df, results = load_data_for_evaluation('city')
	loc = 'city105'
	loc_label = 'Fort Myers, FL'

fig=plt.figure(figsize=(20, 20))

gs=GridSpec(6, 4, height_ratios=[4, 4, 5, 4, 4, 5])

l1=fig.add_subplot(gs[0,:2]) 
l2=fig.add_subplot(gs[0,2:]) 
l3=fig.add_subplot(gs[1,:2]) 
l4=fig.add_subplot(gs[1,2:]) 

v1=fig.add_subplot(gs[2,0]) 
v2=fig.add_subplot(gs[2, 1]) 
v3=fig.add_subplot(gs[2, 2]) 
v4=fig.add_subplot(gs[2, 3]) 

l21=fig.add_subplot(gs[3,:2]) 
l22=fig.add_subplot(gs[3,2:]) 
l23=fig.add_subplot(gs[4,:2]) 
l24=fig.add_subplot(gs[4,2:]) 

v21=fig.add_subplot(gs[5,0]) 
v22=fig.add_subplot(gs[5, 1]) 
v23=fig.add_subplot(gs[5, 2]) 
v24=fig.add_subplot(gs[5, 3]) 


axes = [l1, l2, l3, l4, v1, v2, v3, v4, l21, l22, l23, l24, v21, v22, v23, v24]

for ax in axes:
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)

for ax in [v2, v3, v4, v22, v23, v24]:
	ax.get_yaxis().set_visible(False)
	#ax.spines['left'].set_visible(False)

plot_evolution(l1, {th: results['forecasting_ar'][th] for th in ths}, loc, evolution_palette, 'AR')
plot_evolution(l2, {th: results['forecasting_armulti'][th] for th in ths}, loc, evolution_palette, 'LR')
plot_evolution(l3, {th: results['forecasting_rf'][th] for th in ths}, loc, evolution_palette, 'RF')
plot_evolution(l4, {th: results['forecasting_lstm'][th] for th in ths}, loc, evolution_palette, 'GRU')
plot_evolution(l21, {th: results['nowcasting_ar'][th] for th in ths}, loc, evolution_palette, 'AR')
plot_evolution(l22, {th: results['nowcasting_armulti'][th] for th in ths}, loc, evolution_palette, 'LR')
plot_evolution(l23, {th: results['forecasting_rf'][th] for th in ths}, loc, evolution_palette, 'RF')
plot_evolution(l24, {th: results['nowcasting_lstm'][th] for th in ths}, loc, evolution_palette, 'GRU')

l1.legend(loc='upper left', prop={'size': 12})
l21.legend(loc='upper left', prop={'size': 12})

plot_violins(v1, [results[model][1] for model in ['persistance', 'forecasting_ar', 'forecasting_armulti', 'forecasting_rf', 'forecasting_lstm']], 'rmse', ['Pers.', 'AR', 'LR', 'RF', 'GRU'], palette)
plot_violins(v2, [results[model][2] for model in ['persistance', 'forecasting_ar', 'forecasting_armulti', 'forecasting_rf', 'forecasting_lstm']], 'rmse', ['Pers.', 'AR', 'LR', 'RF', 'GRU'], palette)
plot_violins(v3, [results[model][4] for model in ['persistance', 'forecasting_ar', 'forecasting_armulti', 'forecasting_rf', 'forecasting_lstm']], 'rmse', ['Pers.', 'AR', 'LR', 'RF', 'GRU'], palette)
plot_violins(v4, [results[model][8] for model in ['persistance', 'forecasting_ar', 'forecasting_armulti', 'forecasting_rf', 'forecasting_lstm']], 'rmse', ['Pers.', 'AR', 'LR', 'RF', 'GRU'], palette)

plot_violins(v21, [results[model][1] for model in ['persistance', 'nowcasting_ar', 'nowcasting_armulti', 'nowcasting_rf', 'nowcasting_lstm']], 'rmse', ['Pers.', 'AR', 'LR', 'RF', 'GRU'], palette)
plot_violins(v22, [results[model][2] for model in ['persistance', 'nowcasting_ar', 'nowcasting_armulti', 'nowcasting_rf', 'nowcasting_lstm']], 'rmse', ['Pers.', 'AR', 'LR', 'RF', 'GRU'], palette)
plot_violins(v23, [results[model][4] for model in ['persistance', 'nowcasting_ar', 'nowcasting_armulti', 'nowcasting_rf', 'nowcasting_lstm']], 'rmse', ['Pers.', 'AR', 'LR', 'RF', 'GRU'], palette)
plot_violins(v24, [results[model][8] for model in ['persistance', 'nowcasting_ar', 'nowcasting_armulti', 'nowcasting_rf', 'nowcasting_lstm']], 'rmse', ['Pers.', 'AR', 'LR', 'RF', 'GRU'], palette)


v1.set_title('1 Week Reporting Delay', size='medium')
v2.set_title('2 Week Reporting Delay', size='medium')
v3.set_title('4 Week Reporting Delay', size='medium')
v4.set_title('8 Week Reporting Delay', size='medium')
v21.set_title('1 Week Reporting Delay', size='medium')
v22.set_title('2 Week Reporting Delay', size='medium')
v23.set_title('4 Week Reporting Delay', size='medium')
v24.set_title('8 Week Reporting Delay', size='medium')

v1.set_ylabel('RMSE')
l1.set_ylabel('ILI Incidence')
l3.set_ylabel('ILI Incidence')
v21.set_ylabel('RMSE')
l21.set_ylabel('ILI Incidence')
l23.set_ylabel('ILI Incidence')

loc = plticker.MultipleLocator(base=1.0) # this locator puts ticks at regular intervals
v1.yaxis.set_major_locator(loc)
v21.yaxis.set_major_locator(loc)

plt.tight_layout(rect=[0, 0.06, 1, 0.98])

plt.figtext(0.5, 0.98, 'Predictions of ILI in ' + loc_label + ' Using Only Historical Epi Data', ha='center', va='center', size='large')
plt.figtext(0.5, 0.485, 'Predictions of ILI in ' + loc_label + ' Using Historical Epi Data & GT Data', ha='center', va='center', size='large')
if geogran == 'state':
	plt.figtext(0.5, 0.675, 'Distribution of RMSE Across States (Models Using Only Historical Epi Data)', ha='center', va='center', size='large')
	plt.figtext(0.5, 0.18, 'Distribution of RMSE Across States (Models Using Historical Epi Data & GT Data)', ha='center', va='center', size='large')
else:
	plt.figtext(0.5, 0.675, 'Distribution of RMSE Across Cities (Models Using Only Historical Epi Data)', ha='center', va='center', size='large')
	plt.figtext(0.5, 0.18, 'Distribution of RMSE Across Cities (Models Using Historical Epi Data & GT Data)', ha='center', va='center', size='large')

plt.figtext(0.01, 0.98, 'a', ha='center', va='center', size='x-large', weight='bold')
plt.figtext(0.01, 0.675, 'b', ha='center', va='center', size='x-large', weight='bold')
plt.figtext(0.01, 0.48, 'c', ha='center', va='center', size='x-large', weight='bold')
plt.figtext(0.01, 0.18, 'd', ha='center', va='center', size='x-large', weight='bold')

for ax in [v1, v2, v3, v4]:
	pos1 = ax.get_position() 
	pos2 = [pos1.x0, pos1.y0-0.02,  pos1.width, pos1.height] 
	ax.set_position(pos2) 

for ax in [l21, l22, l23, l24]:
	pos1 = ax.get_position() 
	pos2 = [pos1.x0, pos1.y0-0.04,  pos1.width, pos1.height] 
	ax.set_position(pos2) 

for ax in [v21, v22, v23, v24]:
	pos1 = ax.get_position() 
	pos2 = [pos1.x0, pos1.y0-0.06,  pos1.width, pos1.height] 
	ax.set_position(pos2) 


fig.savefig('figure3_state.png')
