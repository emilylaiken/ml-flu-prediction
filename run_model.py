import sys
import json
import time

from utils import load_flu_cities_subset, load_flu_states, load_trends_states, load_trends_cities

sys.path.insert(0, 'models')
from persistence import persistance
from ar import ar
from ar_with_trends import ar_with_trends
from armulti import ar_multi
from armulti_with_trends import ar_multi_with_trends
from lstm import lstm
from lstm_with_trends import lstm_with_trends

model_lookup = {
	#'persistence':persistance,
	'forecasting_ar':ar, 
	'forecasting_armulti':ar_multi, 
	'forecasting_rf':ar_multi, 
	'forecasting_lstm':lstm, 
	'nowcasting_ar':ar_with_trends,
	'nowcasting_armulti':ar_multi_with_trends,
	'nowcasting_rf':ar_multi_with_trends,
	'nowcasting_lstm':lstm_with_trends
}

start = time.time()

geogran = 'city' # Can be 'state' or 'city'
th = 8 # Can be 1, 2, 4, or 8
model_name = 'nowcasting_lstm' # Can be any of the keys in the model_lookup dictionary above
online_learning = False # Can be True or False
#output_fname = 'new_results_state/' + model_name + '_' + str(th) + '_____'
output_fname = 'bla.json'
if geogran == 'state':
	df = load_flu_states()
	df_trends = load_trends_states()
else:
	df = load_flu_cities_subset()
	df_trends = load_trends_cities()
model = model_lookup[model_name]
n_test = int(sys.argv[1])

# preds = {city:{'dates':[], 'ytrues':[], 'yhats':[], 'coefs':[]} for city in df.columns}
nonlinear = ('rf' in model_name)
if online_learning == True:
	print(n_test)
	for n_test in range(n_test, 0, -1):
		preds = {city:{'dates':[], 'ytrues':[], 'yhats':[], 'coefs':[]} for city in df.columns}
		if 'nowcasting' in model_name:
			if 'multi' in model_name or 'rf' in model_name:
				run, coefs = model(df, df_trends, th, n_test, nonlinear,  False)
			else:
				run, coefs = model(df, df_trends, th, n_test, False)
		else:
			if 'multi' in model_name or 'rf' in model_name:
				run, coefs = model(df, th, n_test, nonlinear,  False)
			else:
				run, coefs = model(df, th, n_test, False)
		for city in df.columns:
			preds[city]['dates'] = run[city][0][0] #.append
			preds[city]['ytrues'] = run[city][1][0]
			preds[city]['yhats'] = run[city][2][0]
			preds[city]['coefs'] = coefs[city]
		with open(output_fname + '_' + str(n_test), 'w') as outfile:
			json.dump(preds, outfile)
else:
	preds = {city:{'dates':[], 'ytrues':[], 'yhats':[], 'coefs':[]} for city in df.columns}
	if 'nowcasting' in model_name:
		if 'multi' in model_name or 'rf' in model_name:
			run, coefs = model(df, df_trends, th, n_test, nonlinear,  False)
		else:
			run, coefs = model(df, df_trends, th, n_test, True) #*******
	else:
		if 'multi' in model_name or 'rf' in model_name:
			run, coefs = model(df, th, n_test, nonlinear,  False)
		else:
			run, coefs = model(df, th, n_test, False)
	for city in df.columns:
		for i in range(len(run[city][0])):
			preds[city]['dates'].append(run[city][0][i])
			preds[city]['ytrues'].append(run[city][1][i])
			preds[city]['yhats'].append(run[city][2][i])
		preds[city]['coefs'].append(coefs[city])
	with open(output_fname + '_' + str(n_test) + '.json', 'w') as outfile:
		json.dump(preds, outfile)

end = time.time()
print(end - start)




