# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 13:16:18 2019

@author: Ashley
"""

# Manuscript Malezieux, Kees, Mulle submitted to Cell Reports
# Description: make dataset for CS and spike/let analysis
# new in this version: take rise time and decay tau from spikes


# %% import modules

import os
import numpy as np
import pandas as pd
from neo import io
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import curve_fit


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

# load data - raw Vm 20 kHz only
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
        Vm = np.copy(bl.segments[0].analogsignals[0].data)
        Vm = Vm[(full_ts >= good_seconds_start[i]) &
                (full_ts < good_seconds_stop[i])]
        Vm = np.squeeze(Vm)
        Vm_ts = full_ts[(full_ts >= good_seconds_start[i]) &
                        (full_ts < good_seconds_stop[i])]
    else:
        sweep_start = int(good_seconds_start[i]/sweep_lenght[i]) 
        sweep_end = int((good_seconds_stop[i]/sweep_lenght[i]))
        sweep_pts = len(bl.segments[0].analogsignals[0].times)
        full_ts = np.zeros(sweep_pts*(sweep_end)) 
        for j in np.arange(sweep_end):
            start_ind = j*sweep_pts
            a = np.squeeze(bl.segments[j].analogsignals[0].times)
            full_ts[start_ind:start_ind+sweep_pts] = a
        Vm_ts = np.zeros(sweep_pts*(sweep_end-sweep_start)) 
        for j in np.arange(sweep_start, sweep_end):
            start_ind = j*sweep_pts - sweep_start*sweep_pts
            a = np.squeeze(bl.segments[j].analogsignals[0].times)
            Vm_ts[start_ind:start_ind+sweep_pts] = a
        Vm = np.zeros(sweep_pts*(sweep_end-sweep_start))
        for l in np.arange(sweep_start, sweep_end):
            start_ind = l*sweep_pts - sweep_start*sweep_pts
            a = np.squeeze(bl.segments[l].analogsignals[0].data)
            Vm[start_ind:start_ind+sweep_pts] = a
        
    # if the file has 'Chan2Hold', load it, if not, create a nan vector
    ds_factor = 10000
    if 'Chan2Hold' in channel_list:
        ind = channel_list.index('Chan2Hold')
        if np.isnan(sweep_lenght[i]):
           Vm_Ih = np.squeeze(np.copy(bl.segments[0].analogsignals[ind].data))
           Vm_Ih = Vm_Ih[(full_ts >= good_seconds_start[i]) &
                         (full_ts < good_seconds_stop[i])] 
        else:
            Vm_Ih = np.zeros(sweep_pts*(sweep_end-sweep_start))
            for f in np.arange(sweep_start, sweep_end):
                start_ind = f*sweep_pts - sweep_start*sweep_pts
                a = np.squeeze(bl.segments[f].analogsignals[ind].data)
                Vm_Ih[start_ind:start_ind+sweep_pts] = a
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

    return Vm_ts, Vm, Vm_Ih_ts, Vm_Ih



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


# definition for detecting complex spikes
def find_CS(Vm_sub, Vm_sub_ts, sp_times, sp_peak_Vm, isi_thresh):
    # find instances where there are spikes with an isi less than isi_thresh
    isi0 = sp_times[0] - Vm_sub_ts[0]
    isi = np.ediff1d(sp_times, to_begin=isi0)
    burst_bool = 1*(isi < isi_thresh)
    # find spikes that are within bursts
    burst_sp = np.where(isi < isi_thresh)[0]
    # find the first spikes in bursts
    burst_sp0 = np.where(np.ediff1d(burst_bool) == 1)[0]
    if burst_sp0.size > 0:
        # remove bursts that only have spikelets
        # simultaneously, save the last spike in the burst
        burst_sp_last = np.zeros(burst_sp0.size, dtype='int64')
        spikelet_thresh = -10  # mV
        non_spikelets = np.full(burst_sp0.size, False)
        for j in np.arange(len(burst_sp0)-1):
            inds = np.append(burst_sp0[j], burst_sp[np.logical_and(burst_sp > burst_sp0[j],
                             burst_sp < burst_sp0[j+1])])
            amps = sp_peak_Vm[inds]
            burst_sp_last[j] = inds[-1]
            if np.any(amps > spikelet_thresh):
                non_spikelets[j] = True
        # special case for the last burst:
        j = len(burst_sp0)-1
        inds = np.append(burst_sp0[j], burst_sp[burst_sp > burst_sp0[j]])
        amps = sp_peak_Vm[inds]
        burst_sp_last[j] = inds[-1]
        if np.any(amps > spikelet_thresh):
                non_spikelets[j] = True
        burst_sp0 = burst_sp0[non_spikelets]
        burst_sp_last = burst_sp_last[non_spikelets]
        # remove doublets from burst_sp0 and burst_sp_last
        burst_sp_last = burst_sp_last[np.isin(burst_sp0+2, burst_sp, assume_unique=True)]
        burst_sp0 = burst_sp0[np.isin(burst_sp0+2, burst_sp, assume_unique=True)]
        # find indices of valleys in Vm_sub
        samp_period = np.round(np.mean(np.diff(Vm_sub_ts)), decimals=3)
        Vm_sub_s = pd.DataFrame(Vm_sub).rolling(3, center=True).mean()
        dVdt = np.ediff1d(Vm_sub_s, to_begin=0)
        dVdt = dVdt/(1000*samp_period)
        Vm_local_min = np.where(np.diff(np.signbit(dVdt))&~np.signbit(np.diff(dVdt)))[0]
        # for each burst_sp0, look in the Vm_sub and determine the start and stop
        # of the subthreshold component of the complex spike
        # CS_start is the same as the first spike in the burst
        CS_start = sp_times[burst_sp0]
        # CS_stop is when the Vm_sub_s has a local minimum after getting within 3mV of baseline
        CS_stop = np.full(burst_sp0.size, np.nan)
        for i in np.arange(burst_sp0.size):
            start_ind = np.searchsorted(Vm_sub_ts, CS_start[i])
            last_ind = np.searchsorted(Vm_sub_ts, sp_times[burst_sp_last[i]])+int(isi_thresh/samp_period)
            # find the first moment the Vm_sub drops below the threshold of the first spike
            # must be at least isi_thresh after the last spike in the burst
            temp_ind = np.where(Vm_sub[last_ind:] < Vm_sub[start_ind]+3)
            if temp_ind[0].size > 0:
                temp_ind = temp_ind[0][0]+last_ind
                # find the first local minimum in the Vm_sub after that
                end_ind = Vm_local_min[np.searchsorted(Vm_local_min, temp_ind)]
                CS_stop[i] = Vm_sub_ts[end_ind]
            else:
                # remove the last burst if the recording is cut during the burst
                CS_start = np.delete(CS_start, i)
                CS_stop = np.delete(CS_stop, i)
        # if there are any CS_start that have the same CS_stop, delete the second
        unique, counts = np.unique(CS_stop, return_counts=True)
        repeat_CS_stop = unique[counts>1]
        for k in np.arange(repeat_CS_stop.size):
            temp_ind = np.where((CS_stop == repeat_CS_stop[k]))[0]
            CS_stop = np.delete(CS_stop, temp_ind[1:])
            CS_start = np.delete(CS_start, temp_ind[1:])
        # if any CS overlap, combine them into one
        overlap_ind = np.where((CS_start[1:] - CS_stop[:-1]) < 0)[0]
        while overlap_ind.size > 0:
            CS_start = np.delete(CS_start, overlap_ind+1)
            CS_stop[overlap_ind] = np.max([CS_stop[overlap_ind], CS_stop[overlap_ind+1]], axis=0)
            CS_stop = np.delete(CS_stop, overlap_ind+1)
            overlap_ind = np.where((CS_start[1:] - CS_stop[:-1]) < 0)[0]
    else:
        CS_start = np.empty(0)
        CS_stop = np.empty(0)
    return CS_start, CS_stop
        
    
# definition to classify spikes
# anything within the bounds of a CS is considered a burst spike/let
# for each CS, save spike information
# divide rest into spikes and spikelets, and divide those into doublets and singles
def class_spikes(CS_start, CS_stop, sp_times, sp_peak_Vm, isi_thresh):
    # find the spikes that occur during each CS
    inds = np.arange(0, sp_times.size, 1, dtype='int64')
    CS = [None]*CS_start.size
    if CS_start.size > 0:
        for i in np.arange(CS_start.size):
            temp_inds = np.where(np.logical_and(sp_times < CS_stop[i],
                                           sp_times >= CS_start[i]))[0]
            CS[i] = temp_inds
        all_CS = np.concatenate(CS)
        inds = np.delete(inds, all_CS)
    # find the spikelets from the non-CS spikes
    temp_inds = np.where(sp_peak_Vm[inds] < -10)[0]
    spikelets = inds[temp_inds]
    inds = np.delete(inds, temp_inds)
    # find the doublets from the remaining spikes
    # note: if a CS was thrown out, some "doublets" will be counted twice
    isi = np.ediff1d(sp_times[inds], to_end = isi_thresh+1)
    temp_inds = np.where(isi < isi_thresh)[0]
    doublets = np.array([inds[temp_inds], inds[temp_inds+1]])
    inds = np.delete(inds, np.concatenate([temp_inds, temp_inds+1]))
    # all the remaining are single spikes
    singles = inds
    return CS, doublets, singles, spikelets
        

# definition to calcuate max rise for each spike
def find_max_rise(Vm, Vm_ts, sp_ind, sp_peak_ind):
    samp_period = np.round(np.mean(np.diff(Vm_ts)), decimals=6)
    max_rise = np.full(sp_ind.size, np.nan)
    for i in np.arange(sp_ind.size):
        try:
            dVdt = np.diff(Vm[sp_ind[i]:sp_peak_ind[i]])/(1000*samp_period)
            max_rise[i] = np.nanmax(dVdt)
        except ValueError:
            max_rise[i] = np.nan
    return max_rise
    

# definition to calculate full-width at half-max for each spike
def find_fwhm(Vm, Vm_ts, sp_ind, sp_end_ind):
    samp_period = np.round(np.mean(np.diff(Vm_ts)), decimals=6)
    fwhm = np.full(sp_ind.size, np.nan)
    for i in np.arange(sp_ind.size):
        try:
            sp_Vm = Vm[sp_ind[i]:sp_end_ind[i]]
            half_max = (np.nanmax(sp_Vm) - sp_Vm[0])/2 + sp_Vm[0]
            inds = np.where(sp_Vm > half_max)[0]
            fwhm[i] = (inds[-1] - inds[0])*samp_period
        except ValueError:
            fwhm[i] = np.nan
    return fwhm


def single_exp(x, a, b):
    return a * np.exp(b * x)


def double_exp(x, a, b, c, d, f):
    return a * np.exp(b * x) + c * np.exp(d * x) + f


# definition to find the decay time constants for spikes
# fits double exponential to the decay, gives time constants
def find_decay(Vm, Vm_ts, sp_peak_ind, sp_end_ind):
    #p0 = [10, -1000, 1, -30, 0]
    decay_tau_1 = np.full(sp_ind.size, np.nan)
    decay_tau_2 = np.full(sp_ind.size, np.nan)
    decay_error = np.full(sp_ind.size, 'ok', dtype='object')
    for j in np.arange(sp_peak_ind.size):
        exp_ts = Vm_ts[sp_peak_ind[j]:sp_end_ind[j]] - Vm_ts[sp_peak_ind[j]]
        trace = Vm[sp_peak_ind[j]+2:sp_end_ind[j]+2]  # shift trace to get off flat part of peak
        trace = trace-trace[-1]
        try:
            #temp_popt, pcov = curve_fit(double_exp, exp_ts, trace, p0)
            #temp_popt, pcov = curve_fit(double_exp, exp_ts, trace)
            temp_popt, pcov = curve_fit(single_exp, exp_ts, trace)
            #decay_tau_1[j] = np.min([-1/temp_popt[1], -1/temp_popt[3]])
            #decay_tau_2[j] = np.max([-1/temp_popt[1], -1/temp_popt[3]])
            decay_tau_1[j] = -1/temp_popt[1]
            decay_tau_2[j] = np.nan
        except RuntimeError:
            decay_tau_1[j] = np.nan
            decay_tau_2[j] = np.nan
            decay_error[j] = 'runtime'
        except ValueError:
            decay_tau_1[j] = np.nan
            decay_tau_2[j] = np.nan
            decay_error[j] = 'value'
        except TypeError:
            decay_tau_1[j] = np.nan
            decay_tau_2[j] = np.nan
            decay_error[j] = 'type'
    return decay_tau_1, decay_tau_2, decay_error



# %% load data
    
# set up list with an empty dictionary for each cell
data = [{'cell_id': 0, 'mouse_id': 0, 'synch': 0}
        for k in np.arange(cells.size)]


for i in np.arange(len(data)):
    # load the raw data from 1 cell at a time
    Vm_ts, Vm, Vm_Ih_ts, Vm_Ih = load_data_MIND(i)
    
    # find the start, peak, and end indices of spikes
    thresh = 5  # V/s (same as 0.25 in old version of spike detection)
    refrac = 1.5  # refractory period, in ms
    peak_win = 3  # window in which there must be a peak, in ms
    sp_ind, sp_peak_ind, sp_end_ind = find_sp_ind_v2(Vm, Vm_ts, thresh, refrac,
                                                     peak_win)

    # make the list of spike times and amplitudes if there are any
    # includes all spikes, spikelets, and spikes within bursts
    sp_times = np.empty(0)
    sp_thresh_Vm = np.empty(0)
    sp_peak_Vm = np.empty(0)
    sp_max_rise = np.empty(0)
    sp_fwhm = np.empty(0)
    sp_rise_time = np.empty(0)
    sp_decay_tau1 = np.empty(0)
    sp_decay_tau2 = np.empty(0)
    decay_error = np.empty(0)
    if sp_ind.size > 0:
        sp_times = Vm_ts[sp_ind]
        sp_thresh_Vm = Vm[sp_ind]
        sp_peak_Vm = Vm[sp_peak_ind]
        sp_max_rise = find_max_rise(Vm, Vm_ts, sp_ind, sp_peak_ind)
        sp_fwhm = find_fwhm(Vm, Vm_ts, sp_ind, sp_end_ind)
        sp_rise_time = Vm_ts[sp_peak_ind] - Vm_ts[sp_ind]
        sp_decay_tau1, sp_decay_tau2, decay_error = find_decay(Vm, Vm_ts, sp_peak_ind, sp_end_ind)
        

    # make a different type of Vm_nosp by removing just the spikes and leaving
    # the subthreshold components of the complex spikes
    if sp_ind.size > 0:
        Vm_nosp = remove_spikes_v2(Vm, sp_ind, sp_end_ind)
    else:
        Vm_nosp = Vm
    
    # downsample and smooth Vm_nosp to 500 Hz to make a subthreshold trace
    ds_factor = 40
    Vm_sub_ts, Vm_sub = ds(Vm_ts, Vm_nosp, ds_factor)

    # detect the complex spikes and save the subthreshold Vm and spike details
    CS_start = np.empty(0)
    CS_stop = np.empty(0)
    CS_ind = np.empty(0)
    doublets_ind = np.empty(0)
    singles_ind = np.empty(0)
    spikelets_ind = np.empty(0)
    isi_thresh = 0.020  # seconds
    if sp_times.size > 0:
        CS_start, CS_stop = find_CS(Vm_sub, Vm_sub_ts, sp_times, sp_peak_Vm, isi_thresh)
        # classify spikes/spikelets as belonging to bursts, doublets, or single spikes
        # gives indices (of sp_ind/sp_times) for the spikes in each category
        a, b, c, d = class_spikes(CS_start, CS_stop, sp_times, sp_peak_Vm, isi_thresh)
        CS_ind = a
        doublets_ind = b
        singles_ind = c
        spikelets_ind = d
        
    # save relevant numbers in data list
    data[i]['cell_id'] = cells[i]
    data[i]['mouse_id'] = mouse_ids[i]
    data[i]['sp_times'] = sp_times
    data[i]['sp_thresh_Vm'] = sp_thresh_Vm
    data[i]['sp_peak_Vm'] = sp_peak_Vm
    data[i]['sp_max_rise'] = sp_max_rise
    data[i]['sp_fwhm'] = sp_fwhm
    data[i]['sp_rise_time'] = sp_rise_time
    data[i]['sp_decay_tau1'] = sp_decay_tau1
    data[i]['sp_decay_tau2'] = sp_decay_tau2
    data[i]['sp_decay_error'] = decay_error
    data[i]['Vm_sub_ts'] = Vm_sub_ts
    data[i]['Vm_sub'] = Vm_sub
    data[i]['CS_start'] = CS_start
    data[i]['CS_stop'] = CS_stop
    data[i]['CS_ind'] = CS_ind
    data[i]['doublets_ind'] = doublets_ind
    data[i]['singles_ind'] = singles_ind
    data[i]['spikelets_ind'] = spikelets_ind



# %% save dictionaries using numpy   
    
    
for i in np.arange(len(data)):
    np.save((r'C:\Users\akees\Documents\Ashley\Analysis\MIND\Python\Datasets\MIND-1\MIND1_v3\CS\cell_' + 
             str(data[i]['cell_id']) + '.npy'), data[i])   
    
    
    
    
