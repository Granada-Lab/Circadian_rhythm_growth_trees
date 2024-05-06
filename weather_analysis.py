#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 16:28:44 2024

@author: nicagutu
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 16:11:39 2024

@author: nicagutu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyboat import WAnalyzer
from pyboat import ensemble_measures as em
import matplotlib.dates as mdates
from datetime import datetime, timedelta

plt.rcParams.update({'font.size': 24})
plt.rcParams['svg.fonttype'] = 'none'

colors = ['tab:blue', 'tab:purple', 'tab:green', 'tab:orange']

dt = 1/6 
lowT = 16
highT = 48
periods = np.linspace(lowT, highT, 500)
wAn = WAnalyzer(periods, dt, time_unit_label='hours')

path = '/Data/'
file = 'produkt_zehn_min_tu_20220915_20240317_07389_2.txt'

df = pd.read_csv(path+file, header=0, index_col=None, delimiter=';')
print(df)

df['MESS_DATUM'] = pd.to_datetime(df['MESS_DATUM'], format='%Y%m%d%H%M')
print(df)

start = pd.Timestamp('2023-05-01 00:00:00')
end = pd.Timestamp('2023-09-15 00:00:00')

dfq = df[(df['MESS_DATUM']>=start) & (df['MESS_DATUM']<=end)]

QN = dfq[dfq.columns[2]]
PP_10 = dfq[dfq.columns[3]]
TT_10 = dfq[dfq.columns[4]]
TM5_10 = dfq[dfq.columns[5]]
RF_10 = dfq[dfq.columns[6]]
TD_10 = dfq[dfq.columns[7]]

# fig, ax = plt.subplots(figsize=(10, 8))
# ax.plot(dfq['MESS_DATUM'], QN)
# plt.xlabel('Day of Year')
# plt.ylabel('Quality level')
# plt.xticks(rotation=90)
# plt.show()

# fig, ax = plt.subplots(figsize=(10, 8))
# ax.plot(dfq['MESS_DATUM'], PP_10)
# plt.xlabel('Day of Year')
# plt.ylabel('Air pressure [hPa]')
# plt.xticks(rotation=90)
# plt.show()


#%%Air temperature
fig, ax = plt.subplots(figsize=(10, 8))
trend = wAn.sinc_smooth(TT_10, T_c=1000)    
ax.plot(dfq['MESS_DATUM'], TT_10)
ax.plot(dfq['MESS_DATUM'], trend, linewidth=5, label='trend')
plt.xlabel('Day of Year')
plt.ylabel('Air temperature [$^\circ$C]')
plt.xticks(rotation=90)
plt.legend(loc='best')
plt.show()
    
wAn.compute_spectrum(TT_10-trend, do_plot=True) 
wAn.get_maxRidge(power_thresh=0, smoothing_wsize=4) 
wAn.draw_Ridge()
wAn.plot_readout(draw_coi=True)

#%%Air temperature at 5cm
fig, ax = plt.subplots(figsize=(10, 8))
trend = wAn.sinc_smooth(TM5_10, T_c=1000)    
ax.plot(dfq['MESS_DATUM'], TM5_10)
ax.plot(dfq['MESS_DATUM'], trend, linewidth=5, label='trend')
plt.xlabel('Day of Year')
plt.ylabel('Air temperature at 5cm [$^\circ$C]')
plt.xticks(rotation=90)
plt.show()

wAn.compute_spectrum(TM5_10-trend, do_plot=True) 
wAn.get_maxRidge(power_thresh=0, smoothing_wsize=4) 
wAn.draw_Ridge()
wAn.plot_readout(draw_coi=True)

#%%Relative humidity
fig, ax = plt.subplots(figsize=(10, 8))
trend = wAn.sinc_smooth(RF_10, T_c=1000)    
ax.plot(dfq['MESS_DATUM'], RF_10)
ax.plot(dfq['MESS_DATUM'], trend, linewidth=5, label='trend')
plt.xlabel('Day of Year')
plt.ylabel('Relative humidity [%]')
plt.xticks(rotation=90)
plt.show()

wAn.compute_spectrum(RF_10-trend, do_plot=True) 
wAn.get_maxRidge(power_thresh=0, smoothing_wsize=4) 
wAn.draw_Ridge()
wAn.plot_readout(draw_coi=True)


#%%Dew point 
fig, ax = plt.subplots(figsize=(10, 8))
trend = wAn.sinc_smooth(TD_10, T_c=1000)    
ax.plot(dfq['MESS_DATUM'], TD_10)
ax.plot(dfq['MESS_DATUM'], trend, linewidth=5, label='trend')
plt.xlabel('Day of Year')
plt.ylabel('Dew point [$^\circ$C]')
plt.xticks(rotation=90)
plt.show()

wAn.compute_spectrum(TD_10-trend, do_plot=True) 
wAn.get_maxRidge(power_thresh=0, smoothing_wsize=4) 
wAn.draw_Ridge()
wAn.plot_readout(draw_coi=True)


