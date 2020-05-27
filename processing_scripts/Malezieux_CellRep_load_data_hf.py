# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 17:31:35 2020

@author: ashley
"""

# Manuscript Malezieux, Kees, Mulle submitted to Cell Reports
# Description: create LFP and Vm traces appropriate for coherence measurements
# Vm_hf - similar to Vm_sub created for CS analysis, but 1250Hz instead of 500Hz


# %% import modules

import os
import numpy as np
import pandas as pd
from neo import io
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import signal


# %% load excel sheet

# load Data_to_Analyze.xlsx to to be able to load data
sheets = ['MIND-1 v3']
cells = np.empty(0)
mouse_ids = np.empty(0)
data_folder = np.empty(0)
ipdep_file = np.empty(0)
stepRa_file = np.empty(0)
ephy_file = np.empty(0)
ephy_file = np.empty(0)
wh_file = np.empty(0)
eye_track = np.empty(0)
sweep_lenght = np.empty(0)
good_seconds_start = np.empty(0)
good_seconds_stop = np.empty(0)
for i in np.arange(len(sheets)):
    data_list = pd.read_excel(r"C:\Users\akees\Documents\Ashley\Analysis\MIND"
                              r"\Data_to_Analyze_exHD.xlsx",
                              sheet_name=sheets[i])
    cells = np.append(cells, data_list.loc[:, 'cell #'].values)
    mouse_ids = np.append(mouse_ids, data_list.loc[:, 'mouse ID'].values)
    data_folder = np.append(data_folder, data_list.loc[:, 'Raw Data Folder'].values)
    ipdep_file = np.append(ipdep_file, data_list.loc[:, 'ip_dep step'].values)
    stepRa_file = np.append(stepRa_file, data_list.loc[:, 'step Ra'].values)
    ephy_file = np.append(ephy_file, data_list.loc[:, 'ephy file'].values)
    wh_file = np.append(wh_file, data_list.loc[:, 'wheel file'].values)
    good_seconds_start = np.append(good_seconds_start,
                                   data_list.loc[:, 'good data start (seconds)'].values)
    good_seconds_stop = np.append(good_seconds_stop,
                                  data_list.loc[:, 'good data end (seconds)'].values)
    sweep_lenght = np.append(sweep_lenght,
                             data_list.loc[:, 'Length of sweeps (s)'].values)
    eye_track = np.append(eye_track,
                          data_list.loc[:, 'Eye tracking file'].values)
    
    
# %% definitions
    
# definition for downsampling
def ds(ts, signal, ds_factor):
    signal_ds = np.mean(np.resize(signal,
                        (int(np.floor(signal.size/ds_factor)), ds_factor)), 1)
    ds_ts = ts[np.arange(int(np.round(ds_factor/2)), ts.size, ds_factor)]
    # trim off last time stamp if necessary
    ds_ts = ds_ts[0:signal_ds.size]
    return ds_ts, signal_ds

# load data - raw Vm 20 kHz and LFP 1250 Hz
# can handle both episodic and gap free
def load_data_MIND(cell_ind):

    i = cell_ind

    # load some Axon data from ABF files
    file_name = os.path.join(data_folder[i], ephy_file[i])
    # r is the name bound to the object created by io.AxonIO
    r = io.AxonIO(filename=file_name)
    # bl is the object that actually has the data, created by read_block
    bl = r.read_block()

    # get list of channel names
    channel_list = []
    for asig in bl.segments[0].analogsignals:
        channel_list.append(asig.name)
    
    if np.isnan(sweep_lenght[i]):
        full_ts = np.copy(bl.segments[0].analogsignals[0].times)
        lfp_raw = np.copy(bl.segments[0].analogsignals[1].data)
        lfp_raw = lfp_raw[(full_ts >= good_seconds_start[i]) &
                          (full_ts < good_seconds_stop[i])]
        lfp_raw = np.squeeze(lfp_raw)
        Vm = np.copy(bl.segments[0].analogsignals[0].data)
        Vm = Vm[(full_ts >= good_seconds_start[i]) &
                (full_ts < good_seconds_stop[i])]
        Vm = np.squeeze(Vm)
        Vm_ts = full_ts[(full_ts >= good_seconds_start[i]) &
                        (full_ts < good_seconds_stop[i])]
    else: 
        sweep_end = len(bl.segments)
        sweep_pts = len(bl.segments[0].analogsignals[0].times)
        full_ts = np.zeros(sweep_pts*sweep_end) 
        for j in np.arange(sweep_end):
            start_ind = j*sweep_pts
            a = np.squeeze(bl.segments[j].analogsignals[0].times)
            full_ts[start_ind:start_ind+sweep_pts] = a
        lfp_raw = np.zeros(sweep_pts*sweep_end)
        for k in np.arange(sweep_end):
            start_ind = k*sweep_pts
            a = np.squeeze(bl.segments[k].analogsignals[1].data)
            lfp_raw[start_ind:start_ind+sweep_pts] = a
        Vm = np.zeros(sweep_pts*sweep_end)
        for l in np.arange(sweep_end):
            start_ind = l*sweep_pts
            a = np.squeeze(bl.segments[l].analogsignals[0].data)
            Vm[start_ind:start_ind+sweep_pts] = a
        # remove the times that we don't want
        lfp_raw = lfp_raw[(full_ts >= good_seconds_start[i]) &
                          (full_ts < good_seconds_stop[i])]
        Vm = Vm[(full_ts >= good_seconds_start[i]) &
                (full_ts < good_seconds_stop[i])]
        Vm_ts = full_ts[(full_ts >= good_seconds_start[i]) &
                        (full_ts < good_seconds_stop[i])]
    
    
    # downsample lfp 16 times to a final frequency of 1250 Hz
    ds_factor = 16
    lfp_ts, lfp_ds = ds(Vm_ts, lfp_raw, ds_factor)
    
    # find the sampling frequency and nyquist of the downsampled LFP
    samp_freq = 1/(lfp_ts[1] - lfp_ts[0])
    nyq = samp_freq/2
    
    # filter the lfp between 0.2 Hz and 300 Hz
    # this algorithm seems to cause no time shift
    # high pass filter
    b, a = signal.butter(4, 0.2/nyq, "high", analog=False)
    lfp_highpass = signal.filtfilt(b, a, lfp_ds)
    # low pass filter
    b, a = signal.butter(4, 300/nyq, "low", analog=False)
    lfp = signal.filtfilt(b, a, lfp_highpass) 
    
    # if the file has 'Chan2Hold', load it, if not, create a nan vector
    ds_factor = 10000
    if 'Chan2Hold' in channel_list:
        ind = channel_list.index('Chan2Hold')
        if np.isnan(sweep_lenght[i]):
           Vm_Ih = np.squeeze(np.copy(bl.segments[0].analogsignals[ind].data))
           Vm_Ih = Vm_Ih[(full_ts >= good_seconds_start[i]) &
                         (full_ts < good_seconds_stop[i])] 
        else:
            Vm_Ih = np.zeros(sweep_pts*sweep_end)
            for f in np.arange(sweep_end):
                start_ind = f*sweep_pts
                a = np.squeeze(bl.segments[f].analogsignals[ind].data)
                Vm_Ih[start_ind:start_ind+sweep_pts] = a
            # keep only the good seconds
            Vm_Ih = Vm_Ih[(full_ts >= good_seconds_start[i]) &
                         (full_ts < good_seconds_stop[i])] 
        # downsample Vm_Ih to 2 Hz (factor of 10000)
        Vm_Ih = np.mean(np.resize(Vm_Ih,
                        (int(np.floor(Vm_Ih.size/ds_factor)), ds_factor)), 1)
        Vm_Ih_ts = Vm_ts[np.arange(0, Vm_ts.size, ds_factor)]
        # trim off last time stamp if necessary
        Vm_Ih_ts = Vm_Ih_ts[0:Vm_Ih.size]
    else:
        Vm_Ih = np.empty(int(Vm_ts.size/ds_factor))
        Vm_Ih[:] = np.nan
        Vm_Ih_ts = Vm_ts[np.arange(0, Vm_ts.size, ds_factor)]
        # trim off last time stamp if necessary
        Vm_Ih_ts = Vm_Ih_ts[0:Vm_Ih.size]

    return Vm_ts, Vm, lfp, lfp_ts, Vm_Ih_ts, Vm_Ih



# find sp_ind, sp_peak_ind, and sp_end_ind at the same time
# thresh is in V/s
# refrac is in ms - blocks detection of multiple spike initiations, peaks
# peak win is in ms - window after the spike inititation when there must be a peak    
def find_sp_ind_v2(Vm, Vm_ts, thresh, refrac, peak_win):
    end_win = 5  # ms after the spike peak to look for the end of the spike
    down_win = refrac  # ms after the spike peak to look for the max down slope 
    # make the dVdt trace
    samp_period = np.round(np.mean(np.diff(Vm_ts)), decimals=6)
    dVdt = np.ediff1d(Vm, to_begin=0)
    dVdt = dVdt/(1000*samp_period)
    # detect when dVdt exceeds the threshold
    dVdt_thresh = np.array(dVdt > thresh, float)
    if sum(dVdt_thresh) == 0:
        # there are no spikes
        sp_ind = np.empty(shape=0)
        sp_peak_ind = np.empty(shape=0)
        sp_end_ind = np.empty(shape=0)
    else:
        # keep just the first index per spike
        sp_ind = np.squeeze(np.where(np.diff(dVdt_thresh) == 1))
        # remove any duplicates of spikes that occur within refractory period
        samp_rate = 1/samp_period
        sp_ind = sp_ind[np.ediff1d(sp_ind,
                        to_begin=samp_rate*refrac/1000+1) >
                        samp_rate*refrac/1000]
        # find the potential spike peaks (tallest within the refractory period)
        dist = refrac/(1000*samp_period)
        sp_peak_ind, _ = signal.find_peaks(Vm, distance=dist, prominence=1)
        # find all the peaks, regardless of the refractory period
        sp_peak_ind_all, _ = signal.find_peaks(Vm, prominence=1)
        # keep only sp_ind when there is a sp_peak_ind within the window
        max_lag = peak_win/(1000*samp_period)
        a = np.searchsorted(sp_peak_ind, sp_ind)
        lags = sp_peak_ind[a] - sp_ind
        sp_ind = sp_ind[lags < max_lag]
        sp_peak_ind = sp_peak_ind[np.searchsorted(sp_peak_ind, sp_ind)]
        # if there are any sp_ind that have the same sp_peak_ind, delete the second
        unique, counts = np.unique(sp_peak_ind, return_counts=True)
        repeat_peak_ind = unique[counts>1]
        for k in np.arange(repeat_peak_ind.size):
            temp_ind = np.where((sp_peak_ind == repeat_peak_ind[k]))[0]
            sp_peak_ind = np.delete(sp_peak_ind, temp_ind[1:])
            sp_ind = np.delete(sp_ind, temp_ind[1:])
        # find the end of spikes
        # first find zero crossings of dVdt where the slope of dVdt is positive
        # this is where Vm has maximum downward slope
        dVdt_min_ind = np.where(np.diff(np.signbit(dVdt))&~np.signbit(np.diff(dVdt)))[0]
        # set the windows over which to look for the max neg.slope and
        # near-zero slope after the peak
        win_ind = samp_rate*end_win/1000
        down_win_ind = samp_rate*down_win/1000
        # find potential ends of spikes and choose the best
        end_ind = np.full(sp_ind.size, np.nan)
        for i in np.arange(sp_ind.size):
            start = int(sp_peak_ind[i])  # start at each spike peak
            # if there is another peak before the next spike, reset the start
            j = np.searchsorted(sp_peak_ind_all, sp_peak_ind[i])
            if np.logical_and((i+1 < sp_peak_ind.size), (j+1 < sp_peak_ind_all.size)):
                b = sp_peak_ind_all[j+1] < sp_peak_ind[i+1]
                c = sp_peak_ind_all[j+1] - sp_peak_ind[i] < down_win_ind
                if np.logical_and(b, c):
                    start = int(sp_peak_ind_all[j+1])
            # set potential stop points to look for max downward and
            # near-zero slopes
            stop1 = int(sp_peak_ind[i]+down_win_ind)
            stop = int(sp_peak_ind[i]+win_ind)
            # reset the stop(s) if another spike starts before then
            if i != sp_ind.size-1:
                if stop > sp_ind[i+1]:
                    stop = sp_ind[i+1]
                if stop1 > sp_ind[i+1]:
                    stop1 = sp_ind[i+1] 
            # find the minimum dVdt (max down slope Vm) between start and stop1
            min_ind = np.argmin(dVdt[start:stop1]) + start
            # find the next Vm zero-slope after the max down slope
            temp_ind = dVdt_min_ind[np.searchsorted(dVdt_min_ind, start)]+1
            if (temp_ind < stop) & (temp_ind > min_ind):
                # if the zero-slope occurs before stop, keep it
                end_ind[i] = temp_ind
            else:
                # if not, find when the Vm slope is closest to zero
                end_ind[i] = np.argmin(np.abs(dVdt[min_ind:stop]))+min_ind
        sp_end_ind = end_ind.astype('int64')
    return sp_ind, sp_peak_ind, sp_end_ind
    


# definition for removing spikes, but keeping subthreshold components of the
# complex spike
def remove_spikes_v2(Vm, sp_ind, sp_end_ind):
    Vm_nosp = np.copy(Vm)
    # linearly interpolate between start and end of each spike
    for i in np.arange(sp_ind.size):
        start = sp_ind[i]
        stop = sp_end_ind[i]
        start_Vm = Vm[start]
        stop_Vm = Vm[stop]
        Vm_nosp[start:stop] = np.interp(np.arange(start, stop, 1),
                                        [start, stop], [start_Vm, stop_Vm])
    return Vm_nosp        


# definition for removing subthreshold spike components from Vm_sub
# for this analysis, want a conservative estimate of average Vm, so will remove
# spikes and underlying depolarizations such as plateau potentials
# search_period is the time (in ms) over which to look for the end of the spike
# VERSION: put nan in place of the values that are taken out
def remove_spikes_sub(Vm_sub_ts, Vm_sub, sp_times, search_period):
    samp_rate = 1/(Vm_sub_ts[1]-Vm_sub_ts[0])
    win = np.array(samp_rate*search_period/1000, dtype='int')
    Vm_nosubsp = np.copy(Vm_sub)
    sp_ind = np.searchsorted(Vm_sub_ts, sp_times)
    for k in np.arange(sp_ind.size):
        # only change Vm if it hasn't been already (i.e. for spikes in bursts)
        if Vm_nosubsp[sp_ind[k]] == Vm_sub[sp_ind[k]]:
            sp_end = np.array(Vm_nosubsp[sp_ind[k]:sp_ind[k]+win] >= Vm_sub[sp_ind[k]],
                              float)
            if np.all(sp_end == 1):
                sp_end = sp_end.size
            else:
                sp_end = np.where(sp_end == 0)[0][0]
            if sp_end > 0:
                sp_end = sp_end+sp_ind[k]
                # no need to interpolate, because start and end = Vm[sp_ind[i]]
                #Vm_nosubsp[sp_ind[k]:sp_end] = Vm_sub[sp_ind[k]]
                Vm_nosubsp[sp_ind[k]:sp_end] = np.nan
    return Vm_nosubsp
    

# %% load data
    
# set up list with an empty dictionary for each cell
data = [{'cell_id': 0, 'mouse_id': 0, 'synch': 0}
        for k in np.arange(cells.size)]


for i in np.arange(len(data)):
    # load the raw data from 1 cell at a time
    Vm_ts, Vm, lfp, lfp_ts, Vm_Ih_ts, Vm_Ih = load_data_MIND(i)
    
    # find the start, peak, and end indices of spikes
    thresh = 5  # V/s (same as 0.25 in old version of spike detection)
    refrac = 1.5  # refractory period, in ms
    peak_win = 3  # window in which there must be a peak, in ms
    sp_ind, sp_peak_ind, sp_end_ind = find_sp_ind_v2(Vm, Vm_ts, thresh, refrac,
                                                     peak_win)       

    # make a different type of Vm_nosp by removing just the spikes and leaving
    # the subthreshold components of the complex spikes
    if sp_ind.size > 0:
        Vm_nosp = remove_spikes_v2(Vm, sp_ind, sp_end_ind)
    else:
        Vm_nosp = Vm
    
    # downsample and filter Vm in the same way as LFP
    ds_factor = 16
    Vm_sub_ts, Vm_sub = ds(Vm_ts, Vm_nosp, ds_factor)
    
    # find the sampling frequency and nyquist of the downsampled LFP
    samp_freq = 1/(Vm_sub_ts[1] - Vm_sub_ts[0])
    nyq = samp_freq/2
    
    # filter the Vm between 0.2 Hz and 300 Hz
    # this algorithm seems to cause no time shift
    # high pass filter
    b, a = signal.butter(4, 0.2/nyq, "high", analog=False)
    Vm_highpass = signal.filtfilt(b, a, Vm_sub)
    # low pass filter
    b, a = signal.butter(4, 300/nyq, "low", analog=False)
    Vm_hf = signal.filtfilt(b, a, Vm_highpass)
    
    # make another Vm trace to use for finding baseline Vm
    search_period = 800  # ms
    if sp_ind.size > 0:
        Vm_hf_nosp = remove_spikes_sub(Vm_sub_ts, Vm_sub, Vm_ts[sp_ind], search_period)
    else:
        Vm_hf_nosp = Vm_sub
        
    # save relevant numbers in data list
    data[i]['cell_id'] = cells[i]
    data[i]['mouse_id'] = mouse_ids[i]
    data[i]['hf_ts'] = lfp_ts
    data[i]['lfp_hf'] = lfp
    data[i]['Vm_hf'] = Vm_hf
    data[i]['Vm_hf_nosp'] = Vm_hf_nosp



# %% save dictionaries using numpy   
    
    
for i in np.arange(len(data)):
    np.save((r'C:\Users\akees\Documents\Ashley\Analysis\MIND\Python\Datasets\MIND-1\MIND1_v3\hf\cell_' + 
             str(data[i]['cell_id']) + '.npy'), data[i])   
    
