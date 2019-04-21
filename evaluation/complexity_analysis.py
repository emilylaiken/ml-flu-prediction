import json
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20

with open('timings/forecasting_ar.json', 'r') as infile:
	ar = json.load(infile)

with open('timings/forecasting_armulti.json', 'r') as infile:
	armulti = json.load(infile)

with open('timings/forecasting_rf.json', 'r') as infile:
	rf = json.load(infile)

with open('timings/lstm100.json', 'r') as infile:
	gru = json.load(infile)


labels = ['AR', 'AR-net LASSO', 'AR-net GRU']
fig, ax = plt.subplots(1, 3, figsize=(20, 5))
ax[0].plot(np.arange(len(ar)), ar, label='AR')
ax[1].plot(np.arange(len(armulti)), armulti, label='AR-net LASSO')
ax[2].plot(np.arange(len(gru)), gru, label='AR-net GRU (100 Training Epochs)')
for a, axis in enumerate(ax):
	axis.set_xlabel('Number of Locations')
	axis.set_title(labels[a])
	axis.spines['right'].set_visible(False)
	axis.spines['top'].set_visible(False)
plt.tight_layout()
plt.subplots_adjust(top=0.8, left=0.04)
plt.suptitle('Model Run Time with Respect to Number of Locations in Dataset', fontsize='x-large')
ax[0].set_ylabel('Time (seconds)')
plt.savefig('timings/complexity2.png')
