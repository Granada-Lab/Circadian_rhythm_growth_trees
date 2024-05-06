#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 11:25:59 2024

@author: nicagutu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyboat import WAnalyzer
import seaborn as sns
from scipy.stats import pearsonr

plt.rcParams.update({'font.size': 24})

dt = 1/6  #sampling intervals
# Circadian range of study between 16-48h
lowT = 16 
highT = 48
periods = np.linspace(lowT, highT, 200)
wAn = WAnalyzer(periods, dt, time_unit_label='hours')

path = '/Data/' #path where your tree growth data is saved
file = 'dendro_Haselberg_23.csv'

df = pd.read_csv(path+file, header=0, index_col=0, delimiter=';')
df.index = pd.to_datetime(df.index)

start = pd.Timestamp('2023-05-01 00:00:00') #starting date
end = pd.Timestamp('2023-09-15 00:00:00') #end date
dfq = df[(df.index>=start) & (df.index<=end)]
df_resampled = dfq.resample('10T').first() #optional: resample date to improve efficiency but conserving resolution

file2 = 'produkt_zehn_min_tu_20220915_20240317_07389_2.txt' #envirnomental conditions cleaned
df2 = pd.read_csv(path+file2, header=0, index_col=None, delimiter=';')
df2['MESS_DATUM'] = pd.to_datetime(df2['MESS_DATUM'], format='%Y%m%d%H%M')
df2.index = df2['MESS_DATUM']
dfq2 = df2[(df2['MESS_DATUM']>=start) & (df2['MESS_DATUM']<=end)]

indexes_to_remove = []
for i in df_resampled.index:
    if i in dfq2.index:
        continue
    else:
        indexes_to_remove.append(i)
indexes_to_remove = pd.Index(indexes_to_remove)
dfq1 = df_resampled.drop(index=indexes_to_remove)

#%%Single tree properties
# list_columns = ['T_Tree','A_Tree','P_Tree','T_TT', 'A_TT','P_TT','T_TT5','A_TT5','P_TT5',
#                 'T_RF','A_RF','P_RF','T_TD','A_TD','P_TD']
# df_all = pd.DataFrame(columns=list_columns)#, index=dfq2.index)

# tree_num = 1
# signal = (dfq1['DR1_0'+str(tree_num)+'_um'])
# detrended = wAn.sinc_detrend(signal, T_c=1000)    
# wAn.compute_spectrum(detrended, do_plot=False) 
# rd = wAn.get_maxRidge(power_thresh=0, smoothing_wsize=4) 
# df_all[df_all.columns[0]] = rd['periods']  
# df_all[df_all.columns[1]] = rd['amplitude']  
# df_all[df_all.columns[2]] = rd['phase']

# for i in range(4):
#     signal = dfq2[dfq2.columns[i+4]]
#     trend = wAn.sinc_smooth(signal, T_c=1000)    
#     wAn.compute_spectrum(signal-trend, do_plot=False) 
#     wAn.get_maxRidge(power_thresh=0, smoothing_wsize=4) 
#     rd = wAn.get_maxRidge(power_thresh=0, smoothing_wsize=4) 
#     df_all[df_all.columns[3*i+3]] = rd['periods']  
#     df_all[df_all.columns[3*i+4]] = rd['amplitude']  
#     df_all[df_all.columns[3*i+5]] = rd['phase']

# df_all.index = dfq2.index
# print(df_all)

# correlation_matrix = df_all.corr()
# plt.figure(figsize=(14, 12))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 12}, cbar_kws={"label": "Correlation Coefficient"})
# plt.show()
    
#%%Average properties trees 

list_columns = ['T_Av_Tree','A_Av_Tree','P_Av_Tree',
                'T_TT', 'A_TT','P_TT','T_TT5','A_TT5','P_TT5',
                'T_RF','A_RF','P_RF','T_TD','A_TD','P_TD']
df_all = pd.DataFrame(columns=list_columns)

# Compute average properties of the trees
df_average_trees_periods = pd.DataFrame()
df_average_trees_amplitude = pd.DataFrame()
df_average_trees_phase = pd.DataFrame()
for i in range(4):   
    tree_num = i+1
    signal = (dfq1['DR1_0'+str(tree_num)+'_um'])
    
    detrended = wAn.sinc_detrend(signal, T_c=1000) #first detrending
    double_detrended = wAn.sinc_detrend(detrended, T_c=100) #second detrending
    
    wAn.compute_spectrum(double_detrended, do_plot=False) #wavelet analysis
    rd = wAn.get_maxRidge(power_thresh=0, smoothing_wsize=4) 
    
    df_average_trees_periods[i] = rd['periods']  
    df_average_trees_amplitude[i] = rd['amplitude']  
    df_average_trees_phase[i] = rd['phase']

df_all[df_all.columns[0]] = df_average_trees_periods.mean(axis=1) #save average period of the 4 trees
df_all[df_all.columns[1]] = df_average_trees_amplitude.mean(axis=1) #save average amplitude of the 4 trees
df_all[df_all.columns[2]] = df_average_trees_phase.mean(axis=1) #save average phase of the 4 trees

# Compute environmental properties 
for i in range(4):
    signal = dfq2[dfq2.columns[i+4]]
    
    detrended = wAn.sinc_detrend(signal, T_c=1000)    

    wAn.compute_spectrum(detrended, do_plot=False)     
    rd = wAn.get_maxRidge(power_thresh=0, smoothing_wsize=4) 
    
    df_all[df_all.columns[3*i+3]] = rd['periods']  
    df_all[df_all.columns[3*i+4]] = rd['amplitude']  
    df_all[df_all.columns[3*i+5]] = rd['phase']

df_all.index = dfq2.index
print(df_all)

# Calculate correlation coefficients
correlation_matrix = df_all.corr()
plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 12}, cbar_kws={'label': 'Correlation Coefficient'})
plt.show()
    
# Calculate p-values
p_values = pd.DataFrame(np.zeros_like(correlation_matrix.values), columns=correlation_matrix.columns, index=correlation_matrix.index)
for i in range(len(correlation_matrix.columns)):
    for j in range(len(correlation_matrix.columns)):
        if i != j:
            corr_coef, p_value = pearsonr(df_all[correlation_matrix.columns[i]], df_all[correlation_matrix.columns[j]])
            p_values.iloc[i, j] = p_value
            
correlation_matrix = df_all.corr()
plt.figure(figsize=(14, 12))
sns.heatmap(p_values, annot=True, cmap='coolwarm', fmt=".4f", annot_kws={"size": 10}, cbar_kws={'label': 'p-values'})
plt.show()            
            
            
