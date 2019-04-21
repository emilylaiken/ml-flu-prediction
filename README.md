# ml-flu-prediction

This repository contains the codebase for the project "Machine Learning for State- and City-level ILI Prediction in the US", with three
particularly relevent sub-projects:

1. Ch. 2 of Emily Aiken's undergraduate thesis for Harvard's computer science concentration, "Machine Learning for Epidemiological 
Prediction"
2. "Interpretability in Data-Driven Epidemiological Forecasting," Emily Aiken and Jonathan Waring's final project for APCOMP221 at Harvard
3. The paper "title tbd" (in progress)

## File Structure
- data
  - Cities_ILI.csv - Raw time-series data of ILI for 316 cities in the US, approximately 2004-2010
  - States_ILI.csv - Raw time-series data of ILI for 37 states in the US, approximately 2011-2017
  - city_trends_subset - Folder containing Google trends time-series for each of 256 keywords for each city available in Google trends 
  that is also in our ILI dataset (180 cities)
  - states_trends - Folder containing Gogle trends time-series for each of 256 keywords for each of 37 states
  - citiesmetadata.csv - Keeps track of which cities are available in both the GT and ILI datasets
  - geocoord.csv - Maps city IDs to their names, states, lat, lon, etc.
- evaluation
  - evaluate_models.py - Code for producing various useful types of figures (violinplots, line plots, etc.)
  - violins.py, splitviolins.py, linechart.py, evolution.py - Example code for producing various useful single figures
  - resultsfigure1.py, resultsfigure2.py, resultsfigure3.py - Code to produce composite figures for paper
  - heatmap.py - Code to generate feature importance heatmaps for different models 
  - makevideo.py - Messy code for producing videos of heatmaps changing over time 
- figures - .png files of figures in paper and supporting materials
- models
  - persistance.py - Code for persistence model
  - ar.py, ar_with_trends.py - Code for AR and ARGO models
  - armulti.py, armulti_with_trends.py - Code for AR-net LR, ARGO-net LR, AR-net RF, ARGO-net RF models
  - lstm.py, lstm_with_trends.py - Code for AR-net GRU and ARGO-net GRU models
- results
  - city - Results by city, time horizon, and model, including feature importances for each model for each week
  - state - Results by state, time horizon, and model, including feature importances for each model for each week
  - no_coef - Contains model runs without feature importances, for quick analyses that don't use heatmaps
- preprocessing.py - Various pre-processing functions (normalization, removing trend time-series with all 0's, etc.)
- run_model.py - Master file to run any of the models from within the model folder
- utils.py - Various useful functions (loading data, loading model runs, etc.)

## Project-Specific Notes

#### Thesis and Paper
The most relevent replication code can be found in the models folder, which contains the code for all models implemented, including 
hyperparameter tuning, etc. Code for evaluation and plots in the paper can be found in the evaluation folder.

#### APCOMP221 Project
TODO:
- Start keeping a coding notebook
- Produce feature importance heatmaps for each week for state-level for GRU (not including any trends data). Save in results folder.
- Cross-influence matrices that show how much each state influences each other state (average the feature importance heatmaps for all the weeks in the evaluation period, then try creating cross-influence matrices by lag (lag 1, lag 2, lag 3, lag 4)
- Clustering of locations based on signatures in feature importance heatmaps (average the heatmaps across all the weeks in the evaluation period, then try this: https://seaborn.pydata.org/generated/seaborn.clustermap.html)
- Project write-up
- Poster

