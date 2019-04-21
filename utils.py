import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error

from preprocessing import remove_zeros

# Returns flu data for 317 US cities for 8 years as a pandas dataframe
def load_flu():
	df = pd.read_csv('data/Cities_ILI.csv', index_col=0)
	df = df.drop(['weeknum', 'season', 'weekyear'], axis=1)
	df.index = pd.to_datetime(df.index)
	return df

# Returns flu data for 317 US cities for 8 years as a pandas dataframe
def load_flu_cities_subset(level=''):
	df = pd.read_csv(level + 'data/Cities_ILI.csv', index_col=0)
	df = df.drop(['weeknum', 'season', 'weekyear'], axis=1)
	joiner = pd.read_csv(level + 'data/citiesmetadata.csv')
	keep = ['city' + str(int(c)) for c in list(joiner['city_id'].values)]
	df = df[keep]
	df.index = pd.to_datetime(df.index)
	df = df[df.index >= pd.to_datetime('1/4/04')]
	return df

# Returns dataframe mapping city ID's to city names and states
def load_geo(level=''):
	geo = pd.read_csv(level + 'data/geocoord.csv')
	geo.index = geo['city_id']
	return geo

def load_dengue():
	df = pd.read_csv('data/dengue.csv', index_col=0)
	df = df.T
	df.index = pd.to_datetime(df.index)
	return df

def load_flu_states(level=''):
	df = pd.read_csv(level + 'data/States_ILI.csv', index_col=0)
	df.index = pd.to_datetime(df.index)
	return df

def load_trends_states():
	epi_df = load_flu_states()
	states = epi_df.columns
	trends = {}
	for state in states:
		df = pd.read_csv('data/states_trends/' + state + '.csv', index_col=0)
		df.columns = [state + '_' + col for col in df.columns]
		df.index = pd.to_datetime(df.index)
		trends[state] = df
	return trends

def load_trends_cities():
	joiner = pd.read_csv('data/citiesmetadata.csv')
	cities = ['city' + str(int(c)) for c in list(joiner['city_id'].values)]
	trends = {}
	for city in cities:
		df = pd.read_csv('data/city_trends_subset/' + city + '.csv', index_col=0)
		df.columns = [city + '_' + col for col in df.columns]
		df.index = pd.to_datetime(df.index)
		trends[city] = df
	return trends

# Load model run from a JSON file
def load_run(fname, num_eval=None):
	try:
		with open(fname, 'r') as infile:
			run = json.load(infile)
			print('loaded file: ' + fname)
			if num_eval is not None:
				for city in run.keys():
					for val in run[city].keys():
						run[city][val] = run[city][val][-num_eval:]
			return run
	except FileNotFoundError:
		print('Could not find file: ' + fname)
		return None

def load_result(geogran, th, model, locs, num_eval=None):
	result = {}
	for loc in locs:
		fname = '../results/' + geogran + '/' + str(th) + '/' + loc + '/' + model + '.json'
		try:
			with open(fname, 'r') as infile:
				run = json.load(infile)
				if num_eval is not None:
					for val in run.keys():
						run[val] = run[val][-num_eval:]
				result[loc] = run
		except FileNotFoundError:
			print('Could not find file: ' + fname)
			return None

# Crop a model run that contains information for a set of cities (in the form of nested dictionaries)
def crop_run(run, min_date, max_date):
	cities = run.keys()
	for city in cities:
		dates = [pd.to_datetime(date) for date in run[city]['dates']]
		if min_date is None:
			eval_ids = [x for x in range(len(dates)) if dates[x] < max_date]
		elif max_date is None:
			eval_ids = [x for x in range(len(dates)) if dates[x] > min_date]
		else:
			eval_ids = [x for x in range(len(dates)) if dates[x] > min_date and dates[x] < max_date]
		for item in run[city].keys():
			run[city][item] = [run[city][item][i] for i in range(len(dates)) if i in eval_ids]
	return run
