import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as plticker

from evaluate_models import load_data_for_evaluation, plot_violins, plot_violins_compare, plot_evolution, evolution_palette, palette

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

fig=plt.figure(figsize=(20, 17))

gs=GridSpec(8, 4, height_ratios=[4, 4, 5, 5, 4, 4, 5, 5])

l1=fig.add_subplot(gs[0,:2]) 
l2=fig.add_subplot(gs[0,2:]) 
l3=fig.add_subplot(gs[1,:2]) 
l4=fig.add_subplot(gs[1,2:]) 

v1=fig.add_subplot(gs[2,0]) 
v2=fig.add_subplot(gs[2, 1]) 
v3=fig.add_subplot(gs[2, 2]) 
v4=fig.add_subplot(gs[2, 3]) 
v5=fig.add_subplot(gs[3,0]) 
v6=fig.add_subplot(gs[3,1]) 
v7=fig.add_subplot(gs[3,2]) 
v8=fig.add_subplot(gs[3,3]) 

l21=fig.add_subplot(gs[4,:2]) 
l22=fig.add_subplot(gs[4,2:]) 
l23=fig.add_subplot(gs[5,:2]) 
l24=fig.add_subplot(gs[5,2:]) 

v21=fig.add_subplot(gs[6,0]) 
v22=fig.add_subplot(gs[6, 1]) 
v23=fig.add_subplot(gs[6, 2]) 
v24=fig.add_subplot(gs[6, 3]) 
v25=fig.add_subplot(gs[7,0]) 
v26=fig.add_subplot(gs[7,1]) 
v27=fig.add_subplot(gs[7,2]) 
v28=fig.add_subplot(gs[7,3]) 


axes = [l1, l2, l3, l4, v1, v2, v3, v4, v5, v6, v7, v8, l21, l22, l23, l24, v21, v22, v23, v24, v25, v26, v27, v28]

for ax in axes:
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)

for ax in [v2, v3, v4, v6, v7, v8, v22, v23, v24, v26, v27, v28]:
	ax.get_yaxis().set_visible(False)
	#ax.spines['left'].set_visible(False)

plot_evolution(l1, {th: results_cities['forecasting_ar'][th] for th in ths}, 'city105', evolution_palette, 'AR (Historical Epi Data Only) Forecasts for Fort Myers, FL')
plot_evolution(l2, {th: results_cities['forecasting_armulti'][th] for th in ths}, 'city105', evolution_palette, 'AR-net LR')
plot_evolution(l3, {th: results_cities['forecasting_rf'][th] for th in ths}, 'city105', evolution_palette, 'AR-net RF')
plot_evolution(l4, {th: results_cities['forecasting_lstm'][th] for th in ths}, 'city105', evolution_palette, 'AR-net GRU')
plot_evolution(l21, {th: results_cities['nowcasting_ar'][th] for th in ths}, 'city105', evolution_palette, 'ARGO')
plot_evolution(l22, {th: results_cities['nowcasting_armulti'][th] for th in ths}, 'city105', evolution_palette, 'ARGO-net LR')
plot_evolution(l23, {th: results_cities['forecasting_rf'][th] for th in ths}, 'city105', evolution_palette, 'AR-net RF')
plot_evolution(l24, {th: results_cities['nowcasting_lstm'][th] for th in ths}, 'city105', evolution_palette, 'ARGO-net GRU')

l1.legend(loc='upper left', prop={'size': 10})
l21.legend(loc='upper left', prop={'size': 10})

l1.set_title('AR Forecasts for Fort Myers, FL (Epi Data Only)')
l2.set_title('AR-net LR Forecasts for Fort Myers, FL (Epi Data Only)')
l3.set_title('AR-net RF Forecasts for Fort Myers, FL (Epi Data Only)')
l4.set_title('AR-net GRU Forecasts for Fort Myers, FL (Epi Data Only)')
l21.set_title('ARGO Nowcasts for Fort Myers, FL (Epi Data & GT Data)')
l22.set_title('ARGO-net LR Nowcasts for Fort Myers, FL (Epi Data & GT Data)')
l23.set_title('ARGO-net RF Nowcasts for Fort Myers, FL (Epi Data & GT Data)')
l24.set_title('ARGO-net GRU Nowcasts for Fort Myers, FL (Epi Data & GT Data)')

plot_violins(v1, [results_states[model][1] for model in ['persistance', 'forecasting_ar', 'forecasting_armulti', 'forecasting_rf', 'forecasting_lstm']], 'rmse', ['Persistence', 'AR', 'AR-net LR', 'AR-net RF', 'AR-net GRU'], palette)
plot_violins(v2, [results_states[model][2] for model in ['persistance', 'forecasting_ar', 'forecasting_armulti', 'forecasting_rf', 'forecasting_lstm']], 'rmse', ['Persistence', 'AR', 'AR-net LR', 'AR-net RF', 'AR-net GRU'], palette)
plot_violins(v3, [results_states[model][4] for model in ['persistance', 'forecasting_ar', 'forecasting_armulti', 'forecasting_rf', 'forecasting_lstm']], 'rmse', ['Persistence', 'AR', 'AR-net LR', 'AR-net RF', 'AR-net GRU'], palette)
plot_violins(v4, [results_states[model][8] for model in ['persistance', 'forecasting_ar', 'forecasting_armulti', 'forecasting_rf', 'forecasting_lstm']], 'rmse', ['Persistence', 'AR', 'AR-net LR', 'AR-net RF', 'AR-net GRU'], palette)

plot_violins(v5, [results_cities[model][1] for model in ['persistance', 'forecasting_ar', 'forecasting_armulti', 'forecasting_rf', 'forecasting_lstm']], 'rmse', ['Persistence', 'AR', 'AR-net LR', 'AR-net RF', 'AR-net GRU'], palette)
plot_violins(v6, [results_cities[model][2] for model in ['persistance', 'forecasting_ar', 'forecasting_armulti', 'forecasting_rf', 'forecasting_lstm']], 'rmse', ['Persistence', 'AR', 'AR-net LR', 'AR-net RF', 'AR-net GRU'], palette)
plot_violins(v7, [results_cities[model][4] for model in ['persistance', 'forecasting_ar', 'forecasting_armulti', 'forecasting_rf', 'forecasting_lstm']], 'rmse', ['Persistence', 'AR', 'AR-net LR', 'AR-net RF', 'AR-net GRU'], palette)
plot_violins(v8, [results_cities[model][8] for model in ['persistance', 'forecasting_ar', 'forecasting_armulti', 'forecasting_rf', 'forecasting_lstm']], 'rmse', ['Persistence', 'AR', 'AR-net LR', 'AR-net RF', 'AR-net GRU'], palette)

plot_violins(v21, [results_states[model][1] for model in ['persistance', 'nowcasting_ar', 'nowcasting_armulti', 'nowcasting_rf', 'nowcasting_lstm']], 'rmse', ['Persistence', 'ARGO', 'ARGO-net LR', 'ARGO-net RF', 'ARGO-net GRU'], palette)
plot_violins(v22, [results_states[model][2] for model in ['persistance', 'nowcasting_ar', 'nowcasting_armulti', 'nowcasting_rf', 'nowcasting_lstm']], 'rmse', ['Persistence', 'ARGO', 'ARGO-net LR', 'ARGO-net RF', 'ARGO-net GRU'], palette)
plot_violins(v23, [results_states[model][4] for model in ['persistance', 'nowcasting_ar', 'nowcasting_armulti', 'nowcasting_rf', 'nowcasting_lstm']], 'rmse', ['Persistence', 'ARGO', 'ARGO-net LR', 'ARGO-net RF', 'ARGO-net GRU'], palette)
plot_violins(v24, [results_states[model][8] for model in ['persistance', 'nowcasting_ar', 'nowcasting_armulti', 'nowcasting_rf', 'nowcasting_lstm']], 'rmse', ['Persistence', 'ARGO', 'ARGO-net LR', 'ARGO-net RF', 'ARGO-net GRU'], palette)

plot_violins(v25, [results_cities[model][1] for model in ['persistance', 'nowcasting_ar', 'nowcasting_armulti', 'nowcasting_rf', 'nowcasting_lstm']], 'rmse', ['Persistence', 'ARGO', 'ARGO-net LR', 'ARGO-net RF', 'ARGO-net GRU'], palette)
plot_violins(v26, [results_cities[model][2] for model in ['persistance', 'nowcasting_ar', 'nowcasting_armulti', 'nowcasting_rf', 'nowcasting_lstm']], 'rmse', ['Persistence', 'ARGO', 'ARGO-net LR', 'ARGO-net RF', 'ARGO-net GRU'], palette)
plot_violins(v27, [results_cities[model][4] for model in ['persistance', 'nowcasting_ar', 'nowcasting_armulti', 'nowcasting_rf', 'nowcasting_lstm']], 'rmse', ['Persistence', 'ARGO', 'ARGO-net LR', 'ARGO-net RF', 'ARGO-net GRU'], palette)
plot_violins(v28, [results_cities[model][8] for model in ['persistance', 'nowcasting_ar', 'nowcasting_armulti', 'nowcasting_rf', 'nowcasting_lstm']], 'rmse', ['Persistence', 'ARGO', 'ARGO-net LR', 'ARGO-net RF', 'ARGO-net GRU'], palette)

for ax in [v1, v2, v3, v4, v5, v6, v7, v8, v21, v22, v23, v24, v25, v26, v27, v28]:
	ax.set_ylim(0, 2)

for ax in [v1, v2, v3, v4, v21, v22, v23, v24]:
	ax.set_xticklabels([])

v1.set_title('1-week Forecasts')
v2.set_title('2-week Forecasts')
v3.set_title('4-week Forecasts')
v4.set_title('8-week Forecasts')
v21.set_title('1-week Nowcasts')
v22.set_title('2-week Nowcasts')
v23.set_title('4-week Nowcasts')
v24.set_title('8-week Nowcasts')

v1.set_ylabel('State-Level RMSE')
v5.set_ylabel('City-Level RMSE')
l1.set_ylabel('ILI Incidence')
l3.set_ylabel('ILI Incidence')
v21.set_ylabel('State-Level RMSE')
v25.set_ylabel('City-Level RMSE')
l21.set_ylabel('ILI Incidence')
l23.set_ylabel('ILI Incidence')

plt.tight_layout(rect=[0, 0, 1, 0.99])


plt.figtext(0.01, 0.98, 'a', ha='center', va='center', size='x-large', weight='bold')
plt.figtext(0.01, 0.755, 'b', ha='center', va='center', size='x-large', weight='bold')
plt.figtext(0.01, 0.49, 'c', ha='center', va='center', size='x-large', weight='bold')
plt.figtext(0.01, 0.265, 'd', ha='center', va='center', size='x-large', weight='bold')

for ax in [v5, v6, v7, v8, v25, v26, v27, v28]:
	pos1 = ax.get_position() 
	pos2 = [pos1.x0, pos1.y0+0.015,  pos1.width, pos1.height] 
	ax.set_position(pos2) 

for ax in [v5, v6, v7, v8]:
	ax.tick_params(axis='x', which='major', labelsize=11)

for ax in [v25, v26, v27, v28]:
	ax.tick_params(axis='x', which='major', labelsize=9)
	ax.tick_params(axis='x', which='minor', labelsize=9)

fig.savefig('bla.png')
