#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 09:17:08 2024

@author: nicagutu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyboat import WAnalyzer

plt.rcParams.update({'font.size': 24})

colors = ['tab:blue', 'tab:purple', 'tab:green', 'tab:orange']

dt = 1/6  #sampling intervals
# Circadian range of study between 16-48h
lowT = 16
highT = 48
periods = np.linspace(lowT, highT, 200)
wAn = WAnalyzer(periods, dt, time_unit_label='hours')

#path where your tree growth data is saved
path = '/Data/'
file = 'dendro_Haselberg_23.csv'

df = pd.read_csv(path+file, header=0, index_col=0, delimiter=';')
df.index = pd.to_datetime(df.index)

df['DayOfYear'] = df.index.dayofyear
df['TimeOfDay'] = df.index.hour + df.index.minute/60

start = pd.Timestamp('2023-May-01 00:00:00')  #starting date
end = pd.Timestamp('2023-Sep-15 00:00:00') #end date
dfq = df[(df.index>=start) & (df.index<=end)]

#optional: resample date to improve efficiency but conserving resolution
df_resampled = dfq.resample('10T').first()

#Raw data of the 4 trees 
fig, ax = plt.subplots(figsize=(10, 8))
for i in range(4):
    ax.plot(df_resampled.index, df_resampled['DR1_0'+str(i+1)+'_um'], label=str(i+1), linewidth=5, color=colors[i])
ax.legend()
plt.xlabel('Day of Year')
plt.ylabel('Raw signal')
plt.xticks(rotation=90)
plt.show()

for i in range(3,4):
    signal = (df_resampled['DR1_0'+str(i+1)+'_um'])
    
    #Computing detrending and double detrending
    detrended = wAn.sinc_detrend(signal, T_c=1000)    
    df_resampled['Detrended'] = detrended
    double_detrended = wAn.sinc_detrend(detrended, T_c=100)
    df_resampled['DoubleDetrended'] = double_detrended
    
    #Raw data
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(df_resampled.index, signal, label='tree '+str(i+1), linewidth=5, color=colors[i])
    ax.legend()
    plt.xlabel('Day of Year')
    plt.ylabel('Raw signal')
    plt.xticks(rotation=90)
    plt.show()

    #First detrending
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(df_resampled.index, detrended, label='tree '+str(i+1), linewidth=5, color=colors[i])
    ax.legend()
    plt.xlabel('Day of Year')
    plt.ylabel('Detrended signal')
    plt.xticks(rotation=90)
    plt.show()

    #Second detrending    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(df_resampled.index, double_detrended, label='tree '+str(i+1), linewidth=5, color=colors[i])
    ax.legend()
    plt.xlabel('Day of Year')
    plt.ylabel('Double detrended signal')
    plt.xticks(rotation=90)
    plt.show()

    #Properties of the first detrended signal
    wAn.compute_spectrum(detrended, do_plot=True) 
    wAn.get_maxRidge(power_thresh=0, smoothing_wsize=4) 
    wAn.draw_Ridge()
    wAn.plot_readout(draw_coi=True)

    #Properties of the double detrended signal   
    wAn.compute_spectrum(double_detrended, do_plot=True) 
    wAn.get_maxRidge(power_thresh=0, smoothing_wsize=4) 
    wAn.draw_Ridge()
    wAn.plot_readout(draw_coi=True)

    #Actogram of the first detrended signal
    pivot_table = df_resampled.pivot_table(index='DayOfYear', columns='TimeOfDay', values='Detrended', aggfunc='mean')
    plt.figure(figsize=(10, 20))  
    plt.imshow(pivot_table, aspect='auto', cmap='coolwarm', origin='lower')
    plt.xlabel('Time of Day [hours]')
    plt.ylabel('Day of Year')
    length = int(len(pivot_table.columns))
    plt.xticks([0,int(length/2),length], [0,12,24])
    plt.colorbar(label='Detrended Value')
    plt.show()

    #Actogram of the double detrended signal    
    pivot_table = df_resampled.pivot_table(index='DayOfYear', columns='TimeOfDay', values='DoubleDetrended', aggfunc='mean')
    plt.figure(figsize=(10, 20))  
    plt.imshow(pivot_table, aspect='auto', cmap='coolwarm', origin='lower')
    plt.xlabel('Time of Day [hours]')
    plt.ylabel('Day of Year')
    length = int(len(pivot_table.columns))
    plt.xticks([0,int(length/2),length], [0,12,24])
    plt.colorbar(label='Double Detrended Value')
    plt.show()


