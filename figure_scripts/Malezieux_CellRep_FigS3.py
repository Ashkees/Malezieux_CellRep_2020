# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 15:05:36 2019

@author: Ashley
"""

# Manuscript Malezieux, Kees, Mulle submitted to Cell Reports
# Figure S3 - Complex spikes
# Description: changes in complex spikes with theta and LIA, plotted separately

# %% import modules

import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
from itertools import compress
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import Divider, Size
from mpl_toolkits.axes_grid1.mpl_axes import Axes



# %% definitions


# bootstrap: one-factor ANOVA-like (for any number of groups):
# is between-group variance bigger than within-group?
def calculate_F(groups_list):
    num_g = len(groups_list)
    box = np.concatenate(groups_list)
    GM = np.nanmedian(box)
    gm = np.zeros(num_g)
    gs = np.zeros(num_g)
    denom = np.zeros(num_g)
    for i in np.arange(num_g):
        gm[i] = np.nanmedian(groups_list[i])
        gs[i] = groups_list[i].size
        denom[i] = np.nansum(np.abs(groups_list[i]-np.nanmedian(groups_list[i])))
    F = (np.sum(gs*np.abs(GM-gm)))/(np.sum(denom))
    return F

# one-way anova for many groups: resampling from big box; only when variances are the same
# returns real_F, p_boot
def boot_anova(groups_list, num_b):
    # compute the real F
    real_F = calculate_F(groups_list)
    faux_F = np.zeros(num_b)
    # compute size and variance of groups
    groups_size = [np.nan]*len(groups_list)
    groups_var = [np.nan]*len(groups_list)
    for g in np.arange(len(groups_list)):
        groups_size[g] = groups_list[g].size
        groups_var[g] = MADAM(groups_list[g], np.nanmedian(groups_list[g]))
    # if the largest variance is more than 2x the smallest, resample within groups
    # demean each group and sample with replacement
    if max(groups_var)/min(groups_var) > 2:
        # subtract the median from each group before resampling
        dm_groups_list = [np.nan] * len(groups_list)
        for g in np.arange(len(groups_list)):
            dm_groups_list[g] = groups_list[g] - np.nanmedian(groups_list[g])
        # shuffle and deal from each group with replacement
        for b in np.arange(num_b):
            # deal into faux groups, each one the same size as in real data
            f_groups_list = [None] * len(groups_list)
            for g in np.arange(len(groups_list)):
                group = dm_groups_list[g]
                resample = group[np.random.randint(0, group.size, size=group.size)]
                f_groups_list[g] = resample
            faux_F[b] = calculate_F(f_groups_list)
        p_boot = np.sum(faux_F > real_F)/num_b
    # if the variances are mostly the same, resample from the big box without replacement
    else:
        box = np.concatenate(groups_list)
        for b in np.arange(num_b):
            np.random.shuffle(box)
            box1 = np.copy(box)
            # deal into fax groups, each one the same size as in real data
            f_groups_list = list()
            for g in np.arange(len(groups_list)):
                f_groups_list.append(box1[0:int(groups_size[g])])
                box1 = box1[int(groups_size[g]):]
            faux_F[b] = calculate_F(f_groups_list)
        p_boot = np.sum(faux_F > real_F)/num_b
    return real_F, p_boot
        

# definition for self_calculated variance (called MADAM??)
# VERSION: accounts for nans when dividing by number of samples
def MADAM(data_pts, descriptor):
    v = np.nansum(np.abs(data_pts-descriptor))/np.sum(~np.isnan(data_pts))
    return v  


def boot_t(t_g0, t_g1, num_b):
    real_d = np.nanmedian(t_g1) - np.nanmedian(t_g0)
    faux_d = np.zeros(num_b)
    box = np.append(t_g0, t_g1)
    for b in np.arange(num_b):
        f_g0 = box[np.random.randint(0, box.size, size=t_g0.size)]
        f_g1 = box[np.random.randint(0, box.size, size=t_g1.size)]
        faux_d[b] = np.nanmedian(f_g1) - np.nanmedian(f_g0)
    p = np.sum(np.abs(faux_d) > np.abs(real_d))/num_b
    return real_d, p


def boot_pair_t(diff, num_b):
    real_d = np.mean(diff)
    faux_d = np.zeros(num_b)
    for b in np.arange(num_b):
        sample = np.random.choice([-1, 1], size = diff.size, replace=True)
        faux_d[b] = np.mean(diff*sample)
    p = np.sum(faux_d<real_d)/num_b
    return real_d, p

# definiton for finding 95% confidence intervals for each bin in histogram
# Version: for a **mean** histogram of **several** histograms
# H_array must be arranged [samples, bins]
def CI_avg_hist(H_array, num_b, CI_perc):
    real_H = np.nanmean(H_array, axis=0)
    faux_H = np.full([H_array.shape[1], num_b], np.nan)
    for b in np.arange(num_b):
        samp = np.random.randint(0, H_array.shape[0], H_array.shape[0])
        faux_H[:, b] = np.nanmean(H_array[samp, :], axis=0)
    CI_low, CI_high = np.nanpercentile(faux_H, [(100-CI_perc)/2, 100-((100-CI_perc)/2)],
                                    axis=1)
    return real_H, CI_high, CI_low
    

# eta = event triggered averages.  CHANGE: nans instead of removing events
def prepare_eta(signal, ts, event_times, win):
    win_npts = [ts[ts < ts[0] + np.abs(win[0])].size,
                ts[ts < ts[0] + np.abs(win[1])].size]
    et_ts = ts[0:np.sum(win_npts)] - ts[0] + win[0]
    et_signal = np.empty(0)
    if event_times.size > 0:
        if signal.ndim == 1:
            et_signal = np.zeros((et_ts.size, event_times.size))
            for i in np.arange(event_times.size):
                if np.logical_or((event_times[i]+win[0]<ts[0]), (event_times[i]+win[1]>ts[-1])):
                    et_signal[:, i] = np.nan*np.ones(et_ts.size)
                else:
                    # find index of closest timestamp to the event time
                    ind = np.argmin(np.abs(ts-event_times[i]))
                    et_signal[:, i] = signal[(ind - win_npts[0]): (ind + win_npts[1])]
        elif signal.ndim == 2:
            et_signal = np.zeros((signal.shape[0], et_ts.size, event_times.size))
            for i in np.arange(event_times.size):
                if np.logical_or((event_times[i]+win[0]<ts[0]), (event_times[i]+win[1]>ts[-1])):
                    et_signal[:, :, i] = np.nan*np.ones([signal.shape[0], et_ts.size])
                else:
                    # find index of closest timestamp to the event time
                    ind = np.argmin(np.abs(ts-event_times[i]))
                    et_signal[:, :, i] = signal[:, (ind - win_npts[0]):
                                                (ind + win_npts[1])]
    return et_signal, et_ts


# eta = event triggered averages
# this code is for point processes, but times instead of inds
def prepare_eta_times(pt_times, event_times, win):
    et_signal = []
    if (pt_times.size > 0) & (event_times.size > 0):
        # find pt_times that occur within window of each event_time
        for i in np.arange(event_times.size):
            ts_section = pt_times[(pt_times > event_times[i] + win[0]) &
                                  (pt_times < event_times[i] + win[1])]
            ts_section = ts_section - event_times[i]
            et_signal.append(ts_section)
    else:
        et_signal = [np.empty(0) for k in np.arange(event_times.size)]
    return et_signal       
        


# eta = event triggered averages: Version: skip events too close to edge
def prepare_eta_skip(signal, ts, event_times, win):
    win_npts = [ts[ts < ts[0] + np.abs(win[0])].size,
                ts[ts < ts[0] + np.abs(win[1])].size]
    et_ts = ts[0:np.sum(win_npts)] - ts[0] + win[0]
    et_signal = np.empty(0)
    if event_times.size > 0:
        # remove any events that are too close to the beginning or end of recording
        if event_times[0]+win[0] < ts[0]:
            event_times = event_times[1:]
        if event_times[-1]+win[1] > ts[-1]:
            event_times = event_times[:-1]
        if signal.ndim == 1:
            et_signal = np.zeros((et_ts.size, event_times.size))
            for i in np.arange(event_times.size):
                # find index of closest timestamp to the event time
                ind = np.argmin(np.abs(ts-event_times[i]))
                et_signal[:, i] = signal[(ind - win_npts[0]): (ind + win_npts[1])]
        elif signal.ndim == 2:
            et_signal = np.zeros((signal.shape[0], et_ts.size, event_times.size))
            for i in np.arange(event_times.size):
                # find index of closest timestamp to the event time
                ind = np.argmin(np.abs(ts-event_times[i]))
                et_signal[:, :, i] = signal[:, (ind - win_npts[0]):
                                            (ind + win_npts[1])]
    return et_signal, et_ts


# %% load data

dataset_folder = (r'C:\Users\akees\Documents\Ashley\Papers\MIND 1\Cell Reports\Dryad upload\Dataset')

cell_files = os.listdir(dataset_folder)
data = [{} for k in np.arange(len(cell_files))]
for i in np.arange(len(cell_files)):
    full_file = os.path.join(dataset_folder, cell_files[i])
    data[i] = np.load(full_file, allow_pickle=True).item()



states = [{'id':'theta', 'bef':-2.5, 'aft':0.5, 'samp_time':2, 't_win':[-3, 3]},
          {'id':'LIA', 'bef':-4, 'aft':-1, 'samp_time':2, 't_win':[-4, 2]}]
ntl = ['nost', 'theta', 'LIA']



# %% process data - for dVm vs dFR analysis

# for each cell, find start and stop times for unlabeled times
for i in np.arange(len(data)):
    state_start = np.concatenate([data[i]['theta_start'], data[i]['LIA_start']])
    state_start = np.sort(state_start)
    state_stop = np.concatenate([data[i]['theta_stop'], data[i]['LIA_stop']])
    state_stop = np.sort(state_stop)
    data[i]['nost_start'] = np.append(data[i]['Vm_ds_ts'][0], state_stop)
    data[i]['nost_stop'] = np.append(state_start, data[i]['Vm_ds_ts'][-1])


# for each cell, make a new spike_times for specifically non-spikelets
for i in np.arange(len(data)):
    data[i]['spike_times'] = np.delete(data[i]['sp_times'],
                                       data[i]['spikelets_ind'])


# for each cell, calculate the isi (inter-spike-interval)
# for true spikes only
for i in np.arange(len(data)):
    if data[i]['spike_times'].size > 0:
        isi0 = data[i]['spike_times'][0] - data[i]['Vm_ds_ts'][0]
        data[i]['isi'] = np.ediff1d(data[i]['spike_times'], to_begin=isi0)
    else:
        data[i]['isi'] = np.empty(0)    


# find the (true) spikes that are within bursts
burst_isi = 0.006  # seconds (Mizuseki 2012 0.006)
for i in np.arange(len(data)):
    if data[i]['spike_times'].size > 0:
        burst_bool = 1*(data[i]['isi'] < burst_isi)
        burst_sp = np.where(data[i]['isi'] < burst_isi)[0]
        burst_sp0 = np.where(np.ediff1d(burst_bool) == 1)[0]
        bursts = [None]*len(burst_sp0)
        if burst_sp0.size > 0:
            for j in np.arange(len(burst_sp0)-1):
                inds = np.append(burst_sp0[j], burst_sp[np.logical_and(burst_sp > burst_sp0[j],
                                 burst_sp < burst_sp0[j+1])])
                bursts[j] = data[i]['spike_times'][inds]
            # special case for the last burst:
            j = len(burst_sp0)-1
            inds = np.append(burst_sp0[j], burst_sp[burst_sp > burst_sp0[j]])
            bursts[j] = data[i]['spike_times'][inds]
        data[i]['bursts'] = bursts
    else:
        data[i]['bursts'] = [None]*0

# add windows triggered by start of some brain states
# collect relative times for (true) spikes, singles, doublets, bursts, and CS
for l in np.arange(len(states)):
    for i in np.arange(len(data)):
        t_Vm, t_ts = prepare_eta(data[i]['Vm_s_ds'], data[i]['Vm_ds_ts'],
                                 data[i][states[l]['id']+'_start'],
                                 states[l]['t_win'])
        t_sp = prepare_eta_times(data[i]['sp_times'],
                                    data[i][states[l]['id']+'_start'],
                                    states[l]['t_win'])
        t_spike = prepare_eta_times(data[i]['spike_times'],
                                    data[i][states[l]['id']+'_start'],
                                    states[l]['t_win'])
        spikelet_times = data[i]['sp_times'][data[i]['spikelets_ind'].astype(int)]
        t_spikelet = prepare_eta_times(spikelet_times,
                                    data[i][states[l]['id']+'_start'],
                                    states[l]['t_win'])
        single_times = data[i]['sp_times'][data[i]['singles_ind'].astype(int)]
        t_single = prepare_eta_times(single_times,
                                     data[i][states[l]['id']+'_start'],
                                     states[l]['t_win'])
        if data[i]['doublets_ind'].size > 0:
            doublet_times = data[i]['sp_times'][data[i]['doublets_ind'][0]]
        else:
            doublet_times = np.empty(0)
        t_doublet = prepare_eta_times(doublet_times,
                                     data[i][states[l]['id']+'_start'],
                                     states[l]['t_win'])
        burst_times = np.array([d[0] for d in data[i]['bursts']])
        t_burst = prepare_eta_times(burst_times,
                                     data[i][states[l]['id']+'_start'],
                                     states[l]['t_win'])
        t_CS = prepare_eta_times(data[i]['CS_start'],
                                     data[i][states[l]['id']+'_start'],
                                     states[l]['t_win'])
        data[i][states[l]['id']+'_Vm'] = t_Vm
        data[i][states[l]['id']+'_sp'] = t_sp  # all spikes and spikelets
        data[i][states[l]['id']+'_spike'] = t_spike  # all spikes (no spikelets)
        data[i][states[l]['id']+'_spikelet'] = t_spikelet
        data[i][states[l]['id']+'_single'] = t_single
        data[i][states[l]['id']+'_doublet'] = t_doublet
        data[i][states[l]['id']+'_burst'] = t_burst
        data[i][states[l]['id']+'_CS'] = t_CS
    states[l]['t_ts'] = t_ts


# add windows triggered by start of some brain states
# collect relative times for (true) spikes, singles, doublets, bursts, and CS
for l in np.arange(len(states)):
    for i in np.arange(len(data)):
        single_times = data[i]['sp_times'][data[i]['singles_ind'].astype(int)]
        if data[i]['doublets_ind'].size > 0:
            doublet_times = np.concatenate(data[i]['sp_times'][data[i]['doublets_ind']])
        else:
            doublet_times = np.empty(0)
        nonCS_times = np.sort(np.concatenate((single_times, doublet_times)))
        t_nonCS = prepare_eta_times(nonCS_times,
                                     data[i][states[l]['id']+'_start'],
                                     states[l]['t_win'])
        if len(data[i]['CS_ind']) > 0:
            CS_times = data[i]['sp_times'][np.concatenate(data[i]['CS_ind'])]
        else:
            CS_times = np.empty(0)
        t_CS = prepare_eta_times(CS_times,
                                     data[i][states[l]['id']+'_start'],
                                     states[l]['t_win'])
        data[i][states[l]['id']+'_CS_spikes'] = t_CS
        data[i][states[l]['id']+'_nonCS_spikes'] = t_nonCS

# for each event in each cell, calculate the CS rate and CS index
for i in np.arange(len(data)):
    for l in np.arange(len(ntl)):
        CS_rate = np.full(data[i][ntl[l]+'_start'].size, np.nan)
        CS_perc = np.full(data[i][ntl[l]+'_start'].size, np.nan)
        for j in np.arange(data[i][ntl[l]+'_start'].size):
            start = data[i][ntl[l]+'_start'][j]
            stop = data[i][ntl[l]+'_stop'][j]
            num_spikes = np.sum(np.logical_and(data[i]['spike_times'] > start,
                                               data[i]['spike_times'] < stop))
            if len(data[i]['CS_ind']) > 0:
                CS_spike_times = data[i]['sp_times'][np.concatenate(data[i]['CS_ind'])]
                num_CS_spikes = np.sum(np.logical_and(CS_spike_times > start,
                                                      CS_spike_times < stop))
                num_CS = np.sum(np.logical_and(data[i]['CS_start'] > start,
                                               data[i]['CS_start'] < stop))
            else:
                num_CS_spikes = 0
                num_CS = 0
            CS_perc[j] = num_CS_spikes/num_spikes
            CS_rate[j] = num_CS/(stop-start)
        data[i][ntl[l]+'_CS_rate'] = CS_rate
        data[i][ntl[l]+'_CS_perc'] = CS_perc
        
            
            






# %% event-based organization for dVm vs dFR
        

# make a dictionary to hold values collapsed over all cells
events = [{}  for k in np.arange(len(states))]
# find Vm0, dVm and significance for each run, excluding when Ih is changed
for l in np.arange(len(states)):
    all_c_p = np.empty(0)
    all_Ih = np.empty(0)
    all_Vm0 = np.empty(0)
    all_dVm = np.empty(0)
    all_dVm_p = np.empty(0)
    for i in np.arange(len(data)):
        samp_freq = 1/(data[i]['Vm_ds_ts'][1] - data[i]['Vm_ds_ts'][0])
        num_ind = int(states[l]['samp_time']*samp_freq)
        # find index of dIh_times
        dIh_ind = data[i]['dIh_times']*samp_freq
        dIh_ind = dIh_ind.astype(int)
        c_p = np.zeros(data[i][states[l]['id']+'_start'].size)
        Ih = np.zeros(data[i][states[l]['id']+'_start'].size)
        Vm0 = np.zeros(data[i][states[l]['id']+'_start'].size)
        dVm = np.zeros(data[i][states[l]['id']+'_start'].size)
        dVm_p = np.zeros(data[i][states[l]['id']+'_start'].size)
        for j in np.arange(data[i][states[l]['id']+'_start'].size):
            # find indices
            bef_ind = int(np.sum(data[i]['Vm_ds_ts'] <
                          (data[i][states[l]['id']+'_start'][j] + states[l]['bef'])))
            aft_ind = int(np.sum(data[i]['Vm_ds_ts'] <
                          (data[i][states[l]['id']+'_start'][j] + states[l]['aft'])))
            # put nan if times are straddling a time when dIh is changed
            dIh_true = np.where((dIh_ind > bef_ind) & (dIh_ind < aft_ind + num_ind))[0]
            if dIh_true.size > 0:
                Ih[j] = np.nan
                Vm0[j] = np.nan
                dVm[j] = np.nan
                dVm_p[j] = np.nan
            else:
                if np.logical_or(l==0, l==1):
                    c_p[j] = data[i][states[l]['id']+'_cell_p']
                else:
                    c_p[j] = data[i]['theta_cell_p']
                Ih_ind = np.searchsorted(data[i]['Vm_Ih_ts'],
                                         data[i][states[l]['id']+'_start'][j])
                Ih[j] = data[i]['Vm_Ih'][Ih_ind]
                # test whether Vm values are significantly different
                # Welch's t-test: normal, unequal variances, independent samp
                t, p = stats.ttest_ind(data[i]['Vm_ds'][bef_ind:bef_ind+num_ind],
                                       data[i]['Vm_ds'][aft_ind:aft_ind+num_ind],
                                       equal_var=False, nan_policy='omit')
                dVm_p[j] = p
                if (np.nanmean(data[i]['Vm_ds'][aft_ind:aft_ind+num_ind]) - 
                    np.nanmean(data[i]['Vm_ds'][bef_ind:bef_ind+num_ind])) > 0:
                    Vm0[j] = np.nanmin(data[i]['Vm_s_ds'][bef_ind:bef_ind+num_ind])
                    dVm[j] = (np.nanmax(data[i]['Vm_s_ds'][aft_ind:aft_ind+num_ind]) - 
                              np.nanmin(data[i]['Vm_s_ds'][bef_ind:bef_ind+num_ind]))
                else:
                    Vm0[j] = np.nanmax(data[i]['Vm_s_ds'][bef_ind:bef_ind+num_ind])
                    dVm[j] = (np.nanmin(data[i]['Vm_s_ds'][aft_ind:aft_ind+num_ind]) - 
                              np.nanmax(data[i]['Vm_s_ds'][bef_ind:bef_ind+num_ind]))
        data[i][states[l]['id']+'_c_p'] = c_p
        data[i][states[l]['id']+'_Ih'] = Ih
        data[i][states[l]['id']+'_Vm0'] = Vm0
        data[i][states[l]['id']+'_dVm'] = dVm
        data[i][states[l]['id']+'_dVm_p'] = dVm_p
        all_c_p = np.append(all_c_p, c_p)
        all_Ih = np.append(all_Ih, Ih)
        all_Vm0 = np.append(all_Vm0, Vm0)
        all_dVm = np.append(all_dVm, dVm)
        all_dVm_p = np.append(all_dVm_p, dVm_p)
    events[l]['c_p'] = all_c_p
    events[l]['Ih'] = all_Ih
    events[l]['Vm0'] = all_Vm0
    events[l]['dVm'] = all_dVm
    events[l]['dVm_p'] = all_dVm_p


# add windows triggered by start of some brain states
for l in np.arange(len(states)):
    for i in np.arange(len(data)):
        t_Vm, t_ts = prepare_eta(data[i]['Vm_s_ds'], data[i]['Vm_ds_ts'],
                                 data[i][states[l]['id']+'_start'],
                                 states[l]['t_win'])
        t_sp = prepare_eta_times(data[i]['sp_times'],
                                 data[i][states[l]['id']+'_start'],
                                 states[l]['t_win'])
        data[i][states[l]['id']+'_Vm'] = t_Vm
        data[i][states[l]['id']+'_sp'] = t_sp
    states[l]['t_ts'] = t_ts 


# add triggered windows to event dictionary
for l in np.arange(len(events)):
    raster_sp = []
    psth_sp = np.empty(0)
    Vm = np.empty((states[l]['t_ts'].shape[0], 0))
    duration = np.empty(0)
    cell_id = np.empty(0)
    for i in np.arange(len(data)):
        cell_psth_sp = np.empty(0)
        if data[i][states[l]['id']+'_start'].size > 0:
            Vm = np.append(Vm, data[i][states[l]['id']+'_Vm'], axis=1)
            duration = np.append(duration, (data[i][states[l]['id']+'_stop'] -
                                     data[i][states[l]['id']+'_start']))
            if isinstance(data[i]['cell_id'], str):
                ind = data[i]['cell_id'].find('_')
                cell_int = int(data[i]['cell_id'][:ind])*np.ones(data[i][states[l]['id']+'_start'].size)
                cell_id = np.append(cell_id, cell_int)
            else:
                cell_int = data[i]['cell_id']*np.ones(data[i][states[l]['id']+'_start'].size)
                cell_id = np.append(cell_id, cell_int)
            for j in np.arange(data[i][states[l]['id']+'_start'].size):
                psth_sp = np.append(psth_sp, data[i][states[l]['id']+'_sp'][j])
                cell_psth_sp = np.append(cell_psth_sp, data[i][states[l]['id']+'_sp'][j])
                raster_sp.append(data[i][states[l]['id']+'_sp'][j])
            data[i][states[l]['id']+'_psth_sp'] = cell_psth_sp
    # remove nans
    no_nan = np.logical_and([~np.isnan(Vm).any(axis=0)],
                            [~np.isnan(events[l]['Vm0'])]).flatten()
    events[l]['Vm'] = Vm[:, no_nan]
    events[l]['cell_id'] = cell_id[no_nan]
    events[l]['duration'] = duration[no_nan]
    events[l]['raster_sp'] = list(compress(raster_sp, no_nan))
    events[l]['c_p'] = events[l]['c_p'][no_nan]
    events[l]['Ih'] = events[l]['Ih'][no_nan]
    events[l]['Vm0'] = events[l]['Vm0'][no_nan]
    events[l]['dVm'] = events[l]['dVm'][no_nan]
    events[l]['dVm_p'] = events[l]['dVm_p'][no_nan]


# %% process data - for CS/burst analysis
        




# for each (true) spike, determine which state it occurs in (and those in no state)
# Version: all spikes, not just those used for spike threshold analysis
for i in np.arange(len(data)):
    nost_sp = np.ones(data[i]['spike_times'].size, dtype=bool)
    for l in np.arange(len(states)):
        state_sp = np.zeros(data[i]['spike_times'].size, dtype=bool)
        for j in np.arange(data[i][states[l]['id']+'_start'].size):
            # find the spikes that occur in that event
            temp_bool = np.all((data[i]['spike_times'] > data[i][states[l]['id']+'_start'][j], 
                                data[i]['spike_times'] < data[i][states[l]['id']+'_stop'][j]),
                               axis=0)
            state_sp = state_sp + temp_bool
        data[i][states[l]['id']+'_spike_bool'] = np.squeeze(state_sp)
        nost_sp = nost_sp*[state_sp == False]
    data[i]['nost_spike_bool'] = np.squeeze(nost_sp)



# for each burst, determine which state it occurs in (and those in no state)
for i in np.arange(len(data)):
    burst_start = np.array([d[0] for d in data[i]['bursts']])
    nost_bst = np.ones(burst_start.size, dtype=bool)
    for l in np.arange(len(states)):
        state_bst = np.zeros(burst_start.size, dtype=bool)
        for j in np.arange(data[i][states[l]['id']+'_start'].size):
            # find the bursts that start during that event
            temp_bool = np.all((burst_start > data[i][states[l]['id']+'_start'][j], 
                                burst_start < data[i][states[l]['id']+'_stop'][j]),
                               axis=0)
            state_bst = state_bst + temp_bool
        data[i][states[l]['id']+'_bst_bool'] = np.squeeze(state_bst)
        nost_bst = nost_bst*[state_bst == False]
    data[i]['nost_bst_bool'] = np.squeeze(nost_bst)
    

# for each cell, determine the % of spikes in bursts for theta, LIA, nost
ntl = ['nost', 'theta', 'LIA']
for i in np.arange(len(data)):
    burst_perc = np.full(3, np.nan)
    sp_times = data[i]['spike_times']
    if len(data[i]['bursts']) > 0:
        burst_times = np.concatenate(data[i]['bursts'])
    else:
        burst_times = 0
    for l in np.arange(len(ntl)):
        total_spikes = 0
        burst_spikes = 0
        for j in np.arange(data[i][ntl[l]+'_start'].size):
            start = data[i][ntl[l]+'_start'][j]
            stop = data[i][ntl[l]+'_stop'][j]
            spikes = np.sum(np.logical_and(sp_times > start, sp_times < stop))
            bursts = np.sum(np.logical_and(burst_times > start, burst_times < stop))
            total_spikes = total_spikes + spikes
            burst_spikes = burst_spikes + bursts
        if total_spikes != 0:
            burst_perc[l] = burst_spikes/total_spikes
    data[i]['burst_perc'] = burst_perc
    

# for each CS, determine which state it occurs in (and those in no state)
for i in np.arange(len(data)):
    nost_CS = np.ones(data[i]['CS_start'].size, dtype=bool)
    for l in np.arange(len(states)):
        state_CS = np.zeros(data[i]['CS_start'].size, dtype=bool)
        for j in np.arange(data[i][states[l]['id']+'_start'].size):
            # find the bursts that start during that event
            temp_bool = np.all((data[i]['CS_start'] > data[i][states[l]['id']+'_start'][j], 
                                data[i]['CS_start'] < data[i][states[l]['id']+'_stop'][j]),
                               axis=0)
            state_CS = state_CS + temp_bool
        data[i][states[l]['id']+'_CS_bool'] = np.squeeze(state_CS)
        nost_CS = nost_CS*[state_CS == False]
    data[i]['nost_CS_bool'] = np.squeeze(nost_CS)



# collect the CS features divided by state
keep_cells = np.where([isinstance(d['cell_id'], int) for d in data])[0]
CS_ntl = [{} for l in np.arange(len(ntl))]
for l in np.arange(len(ntl)):
    num_sp = np.empty(0)
    CS_dur = np.empty(0)
    CS_height_Vm = np.empty(0)
    CS_rel_ahp_Vm = np.empty(0)
    for k in np.arange(keep_cells.size):
        i = keep_cells[k]
        num_sp = np.append(num_sp, np.array([d.size for d in data[i]['CS_ind']])[data[i][ntl[l]+'_CS_bool']])
        CS_dur = np.append(CS_dur, (data[i]['CS_stop'] - data[i]['CS_start'])[data[i][ntl[l]+'_CS_bool']])
        CS_height_Vm = np.append(CS_height_Vm, (data[i]['CS_max_Vm'][data[i][ntl[l]+'_CS_bool']] - 
                                 data[i]['CS_start_Vm'][data[i][ntl[l]+'_CS_bool']]))
        CS_rel_ahp_Vm = np.append(CS_rel_ahp_Vm, (data[i]['CS_stop_Vm'][data[i][ntl[l]+'_CS_bool']] - 
                                 data[i]['CS_start_Vm'][data[i][ntl[l]+'_CS_bool']]))
    CS_ntl[l]['num_sp'] = num_sp
    CS_ntl[l]['CS_dur'] = CS_dur
    CS_ntl[l]['CS_height_Vm'] = CS_height_Vm
    CS_ntl[l]['CS_rel_ahp_Vm'] = CS_rel_ahp_Vm


# %% set figure parameters

# set colors
# states
c_run_theta = [0.398, 0.668, 0.547]
c_nonrun_theta = [0.777, 0.844, 0.773]
c_LIA = [0.863, 0.734, 0.582]
# response type
c_hyp = [0.184, 0.285, 0.430]
c_dep = [0.629, 0.121, 0.047]
c_no = [1, 1, 1]
c_lhyp = [0.62, 0.71, 0.84]
c_ldep = [0.97, 0.71, 0.67]
# dependent variables
c_sp = [0.398, 0.461, 0.703]
c_Vm = [0.398, 0.461, 0.703]
# other
c_lgry = [0.75, 0.75, 0.75]
c_mgry = [0.5, 0.5, 0.5]
c_dgry = [0.25, 0.25, 0.25]
c_wht = [1, 1, 1]
c_blk = [0, 0, 0]
c_bwn = [0.340, 0.242, 0.125]
c_lbwn = [0.645, 0.484, 0.394]
c_grn = [0.148, 0.360, 0.000]

c_dVm = [c_hyp, c_mgry, c_dep]
c_state = [c_mgry, c_run_theta, c_lbwn]
c_state_dark = [c_dgry, c_grn, c_bwn]
c_tl = [c_run_theta, c_lbwn]
c_tnl = [c_run_theta, c_blk, c_lbwn]

# set style defaults
mpl.rcParams['font.size'] = 8
mpl.rcParams['savefig.dpi'] = 1200
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['font.sans-serif'] = "Arial"
mpl.rcParams['font.family'] = "sans-serif"
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.linewidth'] = 1
mpl.rcParams['xtick.major.size'] = 4
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['ytick.major.size'] = 4
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['boxplot.whiskerprops.linestyle'] = '-'
mpl.rcParams['patch.force_edgecolor'] = True
mpl.rcParams['patch.facecolor'] = 'b'


# set figure output folder
fig_folder = r'C:\Users\akees\Documents\Ashley\Figures\2020-05_Paper_MIND1\FigS3'

# set which states to plot
## all states
#d_l = [0, 1, 2]
# theta only
d_l = [0, 1]
## LIA only
#d_l = [0, 2]



# %% make hist isi figure


keep_cells = [isinstance(d['cell_id'], int) for d in data]
theta_cell_p = np.array([d['theta_cell_p'] for d in data])[keep_cells]
LIA_cell_p = np.array([d['LIA_cell_p'] for d in data])[keep_cells]

c_state_hist = [c_mgry, c_grn, c_bwn]
c_state_fill = [c_lgry, c_run_theta, c_lbwn]

# prep numbers for mean hist isi - divided between states
bins = np.arange(0, 200, 1)
H = np.full([len(data), bins.size-1, len(ntl)], np.nan)
for i in np.arange(len(data)):
    for l in np.arange(len(ntl)):
        H[i, :, l] = np.histogram(1000*data[i]['isi'][data[i][ntl[l]+'_spike_bool']],
                                  bins=bins, density=True)[0]
# remove extra recordings from cells
H = H[keep_cells, :, :]
# define the 95% CI for each bin by randomly selecting (with replacement) over cells
H_mean = np.full([H.shape[1], H.shape[2]], np.nan)
H_CI_high = np.full([H.shape[1], H.shape[2]], np.nan)
H_CI_low = np.full([H.shape[1], H.shape[2]], np.nan)
CI_perc = 95
num_b = 1000
for l in np.arange(len(ntl)):
    real_H, CI_high, CI_low = CI_avg_hist(H[:, :, l], num_b, CI_perc)
    H_mean[:, l] = real_H
    H_CI_high[:, l] = CI_high
    H_CI_low[:, l] = CI_low
# plot the mean hist isi
fig, ax = plt.subplots(1, figsize=[4.5, 2.2])
for l in d_l:
    ax.plot(bins[:-1], H_mean[:, l], color=c_state_hist[l], zorder=2)
    ax.fill_between(bins[:-1], H_CI_low[:, l], H_CI_high[:, l],
                    facecolor=c_state_fill[l], linewidth=0, zorder=1, alpha=0.25)
#ax.axvline(6, color=c_blk, linestyle='--')
#ax.set_xlim([0, 200])
ax.set_ylim([0, 0.27])
ax.set_yticks([0, 0.05, 0.1, 0.15, 0.2, 0.25]) 
ax.set_yticklabels([0, '', 0.1, '', 0.2, ''])
ax.set_ylabel('proportion')
ax.set_xscale('log')
ax.set_xlim([1, 100])
ax.set_xlabel('inter-spike interval (ms)')
fig.tight_layout()
plt.savefig(os.path.join(fig_folder, ntl[d_l[-1]]+'_hist_isi.png'), transparent=True)


# do the stats for the above figure
S = np.full([len(data), len(ntl)], np.nan)
for i in np.arange(len(data)):
    for l in np.arange(len(ntl)):
        isi = 1000*data[i]['isi'][data[i][ntl[l]+'_spike_bool']]
        if isi.size > 10:
            S[i, l] = np.nanmedian(isi)
# remove extra recordings from cells        
S = S[keep_cells, :]

# do the paired boot stats
num_b = 1000
p = np.full(len(ntl) - 1, np.nan)
d = np.full(len(ntl) - 1, np.nan)
for l in np.arange(len(ntl) - 1):
    dif = S[:, l+1] - S[:, 0]
    # remove nans
    dif = dif[~np.isnan(dif)]
    d[l], p[l] = boot_pair_t(dif, num_b)
    print(dif.size)
print(d)
print(p)

# do the paired boot stats for theta hyp and LIA dep cells only
num_b = 1000
# theta hyp cells
dif = S[:, 1][theta_cell_p < 0.05] - S[:, 0][theta_cell_p < 0.05]
# remove nans
dif = dif[~np.isnan(dif)]
d, p = boot_pair_t(dif, num_b)
print(dif.size)
print(d)
print(p)
# LIA dep cells
dif = S[:, 2][LIA_cell_p > 0.95] - S[:, 0][LIA_cell_p > 0.95]
# remove nans
dif = dif[~np.isnan(dif)]
d, p = boot_pair_t(dif, num_b)
print(dif.size)
print(d)
print(p)

# descriptive numbers
l = 1
np.sum(~np.isnan(S[:, l]))
np.nanmedian(S[:, l])
np.nanstd(S[:, l])
MADAM(S[:, l], np.nanmedian(S[:, l]))


# %% make CS features figures - number of spikes per CS

keep_cells = [isinstance(d['cell_id'], int) for d in data]
theta_cell_p = np.array([d['theta_cell_p'] for d in data])[keep_cells]
LIA_cell_p = np.array([d['LIA_cell_p'] for d in data])[keep_cells]

c_state_hist = [c_mgry, c_grn, c_bwn]
c_state_fill = [c_lgry, c_run_theta, c_lbwn]

# prep numbers for mean hist of # spikes in CS - divided between states
bins = np.arange(0.5, 51.5, 1)
H = np.full([len(data), bins.size-1, len(ntl)], np.nan)
for i in np.arange(len(data)):
    for l in np.arange(len(ntl)):
        num_sp = np.array([d.size for d in data[i]['CS_ind']])[data[i][ntl[l]+'_CS_bool']]
        H[i, :, l] = np.histogram(num_sp, bins=bins)[0]
        # normalize to total number of CS
        H[i, :, l] = H[i, :, l]/np.sum(H[i, :, l])
# remove extra recordings from cells
H = H[keep_cells, :, :]
# define the 95% CI for each bin by randomly selecting (with replacement) over cells
H_mean = np.full([H.shape[1], H.shape[2]], np.nan)
H_CI_high = np.full([H.shape[1], H.shape[2]], np.nan)
H_CI_low = np.full([H.shape[1], H.shape[2]], np.nan)
CI_perc = 95
num_b = 1000
for l in np.arange(len(ntl)):
    real_H, CI_high, CI_low = CI_avg_hist(H[:, :, l], num_b, CI_perc)
    H_mean[:, l] = real_H
    H_CI_high[:, l] = CI_high
    H_CI_low[:, l] = CI_low
# plot the mean hist isi
# create a figure with axes of defined size
fig = plt.figure(figsize=[2, 2])
# The first items are for padding and the second items are for the axes.
# sizes are in inch.
h = [Size.Fixed(0.5), Size.Fixed(1.2)]
v = [Size.Fixed(0.5), Size.Fixed(1.2)]
divider = Divider(fig, (0.0, 0.0, 1., 1.), h, v, aspect=False)
ax = Axes(fig, divider.get_position())
ax.set_axes_locator(divider.new_locator(nx=1, ny=1))
fig.add_axes(ax)
#for l in np.arange(len(ntl)):
for l in d_l:
    ax.plot(np.arange(1, bins.size), H_mean[:, l], color=c_state_hist[l], zorder=2)
    ax.fill_between(np.arange(1, bins.size), H_CI_low[:, l], H_CI_high[:, l],
                    facecolor=c_state_fill[l], linewidth=0, zorder=1, alpha=0.25)
ax.set_ylim([0, 0.4])
ax.set_yticks([0, 0.2, 0.4])
ax.set_yticklabels([0, '', 0.4])
ax.set_xlim([3, 12])
ax.set_xticks([3, 6, 9, 12])
ax.set_xlabel('number of spikes')
ax.set_ylabel('proportion')
plt.savefig(os.path.join(fig_folder, ntl[d_l[-1]]+'_CS_num_spikes.png'), transparent=True)



# do the stats for the above figure
S = np.full([len(data), len(ntl)], np.nan)
for i in np.arange(len(data)):
    for l in np.arange(len(ntl)):
        num_sp = np.array([d.size for d in data[i]['CS_ind']])[data[i][ntl[l]+'_CS_bool']]
        if num_sp.size > 0:
            #S[i, l] = stats.mode(num_sp)[0]
            S[i, l] = np.nanmedian(num_sp)
# remove extra recordings from cells        
S = S[keep_cells, :]

# do the paired boot stats
num_b = 1000
p = np.full(len(ntl) - 1, np.nan)
d = np.full(len(ntl) - 1, np.nan)
for l in np.arange(len(ntl) - 1):
    dif = S[:, l+1] - S[:, 0]
    # remove nans
    dif = dif[~np.isnan(dif)]
    d[l], p[l] = boot_pair_t(dif, num_b)
    print(dif.size)
print(d)
print(p)

# do the paired boot stats for theta hyp and LIA dep cells only
num_b = 1000
# theta hyp cells
dif = S[:, 1][theta_cell_p < 0.05] - S[:, 0][theta_cell_p < 0.05]
# remove nans
dif = dif[~np.isnan(dif)]
d, p = boot_pair_t(dif, num_b)
print(dif.size)
print(d)
print(p)
# LIA dep cells
dif = S[:, 2][LIA_cell_p > 0.95] - S[:, 0][LIA_cell_p > 0.95]
# remove nans
dif = dif[~np.isnan(dif)]
d, p = boot_pair_t(dif, num_b)
print(dif.size)
print(d)
print(p)

# descriptive numbers
l = 1
np.sum(~np.isnan(S[:, l]))
np.nanmedian(S[:, l])
np.nanstd(S[:, l])
MADAM(S[:, l], np.nanmedian(S[:, l]))



# %% make CS features figures - CS duration

# prep numbers for mean hist CS duration - divided between states
bins = np.arange(0, 0.5, 0.02)
H = np.full([len(data), bins.size-1, len(ntl)], np.nan)
for i in np.arange(len(data)):
    for l in np.arange(len(ntl)):
        CS_dur = (data[i]['CS_stop'] - data[i]['CS_start'])[data[i][ntl[l]+'_CS_bool']]
        H[i, :, l] = np.histogram(CS_dur, bins=bins)[0]
        # normalize to total number of CS
        H[i, :, l] = H[i, :, l]/np.sum(H[i, :, l])
# define the 95% CI for each bin by randomly selecting (with replacement) over cells
H_mean = np.full([H.shape[1], H.shape[2]], np.nan)
H_CI_high = np.full([H.shape[1], H.shape[2]], np.nan)
H_CI_low = np.full([H.shape[1], H.shape[2]], np.nan)
CI_perc = 95
num_b = 1000
for l in np.arange(len(ntl)):
    real_H, CI_high, CI_low = CI_avg_hist(H[:, :, l], num_b, CI_perc)
    H_mean[:, l] = real_H
    H_CI_high[:, l] = CI_high
    H_CI_low[:, l] = CI_low
# plot the mean hist CS duration
# create a figure with axes of defined size
fig = plt.figure(figsize=[2, 2])
# The first items are for padding and the second items are for the axes.
# sizes are in inch.
h = [Size.Fixed(0.5), Size.Fixed(1.2)]
v = [Size.Fixed(0.5), Size.Fixed(1.2)]
divider = Divider(fig, (0.0, 0.0, 1., 1.), h, v, aspect=False)
ax = Axes(fig, divider.get_position())
ax.set_axes_locator(divider.new_locator(nx=1, ny=1))
fig.add_axes(ax)
for l in d_l:
    ax.plot(bins[:-1], H_mean[:, l], color=c_state_hist[l], zorder=2)
    ax.fill_between(bins[:-1], H_CI_low[:, l], H_CI_high[:, l],
                    facecolor=c_state_fill[l], linewidth=0, zorder=1, alpha=0.25)
ax.set_xlim([0, 0.2]) 
ax.set_ylim([0, 0.5])
ax.set_xticks([0, 0.1, 0.2])
ax.set_yticks([0, 0.25, 0.5])
ax.set_yticklabels([0, '', 0.5])
ax.set_xlabel('duration (ms)')
plt.savefig(os.path.join(fig_folder, ntl[d_l[-1]]+'_CS_dur.png'), transparent=True)


# do the stats for the above figure
S = np.full([len(data), len(ntl)], np.nan)
for i in np.arange(len(data)):
    for l in np.arange(len(ntl)):
        CS_dur = 1000*(data[i]['CS_stop'] - data[i]['CS_start'])[data[i][ntl[l]+'_CS_bool']]
        if CS_dur.size > 0:
            S[i, l] = np.nanmedian(CS_dur)
# remove extra recordings from cells        
S = S[keep_cells, :]

## do the kruskall-wallace
#H, p_kw = stats.kruskal(S[:, 0], S[:, 1], S[:, 2], nan_policy='omit')

# do the paired boot stats
num_b = 1000
p = np.full(len(ntl) - 1, np.nan)
d = np.full(len(ntl) - 1, np.nan)
for l in np.arange(len(ntl) - 1):
    dif = S[:, l+1] - S[:, 0]
    # remove nans
    dif = dif[~np.isnan(dif)]
    d[l], p[l] = boot_pair_t(dif, num_b)
    print(dif.size)
print(d)
print(p)

# do the paired boot stats for theta hyp and LIA dep cells only
num_b = 1000
# theta hyp cells
dif = S[:, 1][theta_cell_p < 0.05] - S[:, 0][theta_cell_p < 0.05]
# remove nans
dif = dif[~np.isnan(dif)]
d, p = boot_pair_t(dif, num_b)
print(dif.size)
print(d)
print(p)
# LIA dep cells
dif = S[:, 2][LIA_cell_p > 0.95] - S[:, 0][LIA_cell_p > 0.95]
# remove nans
dif = dif[~np.isnan(dif)]
d, p = boot_pair_t(dif, num_b)
print(dif.size)
print(d)
print(p)
# LIA nondep cells
dif = S[:, 2][LIA_cell_p < 0.95] - S[:, 0][LIA_cell_p < 0.95]
# remove nans
dif = dif[~np.isnan(dif)]
d, p = boot_pair_t(dif, num_b)
print(dif.size)
print(d)
print(p)

# descriptive numbers
l = 1
np.sum(~np.isnan(S[:, l]))
np.nanmedian(S[:, l])
np.nanstd(S[:, l])
MADAM(S[:, l], np.nanmedian(S[:, l]))



# %% make CS features figures - subthreshold depolarization during CS

# prep numbers for mean hist CS max-start Vm - divided between states
bins = np.arange(0, 40, 2)
H = np.full([len(data), bins.size-1, len(ntl)], np.nan)
for i in np.arange(len(data)):
    for l in np.arange(len(ntl)):
        CS_height_Vm = (data[i]['CS_max_Vm'][data[i][ntl[l]+'_CS_bool']] - 
                        data[i]['CS_start_Vm'][data[i][ntl[l]+'_CS_bool']])
        H[i, :, l] = np.histogram(CS_height_Vm, bins=bins)[0]
        # normalize to total number of CS
        H[i, :, l] = H[i, :, l]/np.sum(H[i, :, l])
# remove extra recordings from cells
H = H[keep_cells, :, :]
# define the 95% CI for each bin by randomly selecting (with replacement) over cells
H_mean = np.full([H.shape[1], H.shape[2]], np.nan)
H_CI_high = np.full([H.shape[1], H.shape[2]], np.nan)
H_CI_low = np.full([H.shape[1], H.shape[2]], np.nan)
CI_perc = 95
num_b = 1000
for l in np.arange(len(ntl)):
    real_H, CI_high, CI_low = CI_avg_hist(H[:, :, l], num_b, CI_perc)
    H_mean[:, l] = real_H
    H_CI_high[:, l] = CI_high
    H_CI_low[:, l] = CI_low
# plot the mean hist isi
# create a figure with axes of defined size
fig = plt.figure(figsize=[2, 2])
# The first items are for padding and the second items are for the axes.
# sizes are in inch.
h = [Size.Fixed(0.5), Size.Fixed(1.2)]
v = [Size.Fixed(0.5), Size.Fixed(1.2)]
divider = Divider(fig, (0.0, 0.0, 1., 1.), h, v, aspect=False)
ax = Axes(fig, divider.get_position())
ax.set_axes_locator(divider.new_locator(nx=1, ny=1))
fig.add_axes(ax)
for l in d_l:
    ax.plot(bins[:-1], H_mean[:, l], color=c_state_hist[l], zorder=2)
    ax.fill_between(bins[:-1], H_CI_low[:, l], H_CI_high[:, l],
                    facecolor=c_state_fill[l], linewidth=0, zorder=1, alpha=0.25)
#ax.set_xlabel('CS height (mV)')
#ax.set_ylabel('proportion of CS')
ax.set_xlim([0, 35])
ax.set_xticks([0, 10, 20, 30])
ax.set_xlabel('subthreshold depolarization (mV)')
ax.set_ylim([0, 0.3])
ax.set_yticks([0, 0.1, 0.2, 0.3])
ax.set_yticklabels([0, '', '', 0.3])
plt.savefig(os.path.join(fig_folder, ntl[d_l[-1]]+'_CS_height.png'), transparent=True)



# do the stats for the above figure
S = np.full([len(data), len(ntl)], np.nan)
for i in np.arange(len(data)):
    for l in np.arange(len(ntl)):
        CS_height_Vm = (data[i]['CS_max_Vm'][data[i][ntl[l]+'_CS_bool']] - 
                        data[i]['CS_start_Vm'][data[i][ntl[l]+'_CS_bool']])
        if CS_height_Vm.size > 0:
            S[i, l] = np.nanmedian(CS_height_Vm)
# remove extra recordings from cells        
S = S[keep_cells, :]

# do the paired boot stats
num_b = 1000
p = np.full(len(ntl) - 1, np.nan)
d = np.full(len(ntl) - 1, np.nan)
for l in np.arange(len(ntl) - 1):
    dif = S[:, l+1] - S[:, 0]
    # remove nans
    dif = dif[~np.isnan(dif)]
    d[l], p[l] = boot_pair_t(dif, num_b)
    print(dif.size)
print(d)
print(p)

# do the paired boot stats for theta hyp and LIA dep cells only
num_b = 1000
# theta hyp cells
dif = S[:, 1][theta_cell_p < 0.05] - S[:, 0][theta_cell_p < 0.05]
# remove nans
dif = dif[~np.isnan(dif)]
d, p = boot_pair_t(dif, num_b)
print(dif.size)
print(d)
print(p)
# LIA dep cells
dif = S[:, 2][LIA_cell_p > 0.95] - S[:, 0][LIA_cell_p > 0.95]
# remove nans
dif = dif[~np.isnan(dif)]
d, p = boot_pair_t(dif, num_b)
print(dif.size)
print(d)
print(p)
## LIA nondep cells
#dif = S[:, 2][LIA_cell_p < 0.95] - S[:, 0][LIA_cell_p < 0.95]
## remove nans
#dif = dif[~np.isnan(dif)]
#d, p = boot_pair_t(dif, num_b)
#print(dif.size)
#print(d)
#print(p)

# descriptive numbers
l = 1
np.sum(~np.isnan(S[:, l]))
np.nanmedian(S[:, l])
np.nanstd(S[:, l])
MADAM(S[:, l], np.nanmedian(S[:, l]))


# %% make CS features figures - after-CS hyperpolarization

# prep numbers for mean hist CS relative ahp - divided between states
bins = np.arange(-25, 10, 2)
H = np.full([len(data), bins.size-1, len(ntl)], np.nan)
for i in np.arange(len(data)):
    for l in np.arange(len(ntl)):
        CS_rel_ahp_Vm = (data[i]['CS_stop_Vm'][data[i][ntl[l]+'_CS_bool']] - 
                        data[i]['CS_start_Vm'][data[i][ntl[l]+'_CS_bool']])
        H[i, :, l] = np.histogram(CS_rel_ahp_Vm, bins=bins)[0]
        # normalize to total number of CS
        H[i, :, l] = H[i, :, l]/np.sum(H[i, :, l])
# remove extra recordings from cells
H = H[keep_cells, :, :]
# define the 95% CI for each bin by randomly selecting (with replacement) over cells
H_mean = np.full([H.shape[1], H.shape[2]], np.nan)
H_CI_high = np.full([H.shape[1], H.shape[2]], np.nan)
H_CI_low = np.full([H.shape[1], H.shape[2]], np.nan)
CI_perc = 95
num_b = 1000
for l in np.arange(len(ntl)):
    real_H, CI_high, CI_low = CI_avg_hist(H[:, :, l], num_b, CI_perc)
    H_mean[:, l] = real_H
    H_CI_high[:, l] = CI_high
    H_CI_low[:, l] = CI_low
# plot the mean hist isi
# create a figure with axes of defined size
fig = plt.figure(figsize=[2, 2])
# The first items are for padding and the second items are for the axes.
# sizes are in inch.
h = [Size.Fixed(0.5), Size.Fixed(1.2)]
v = [Size.Fixed(0.5), Size.Fixed(1.2)]
divider = Divider(fig, (0.0, 0.0, 1., 1.), h, v, aspect=False)
ax = Axes(fig, divider.get_position())
ax.set_axes_locator(divider.new_locator(nx=1, ny=1))
fig.add_axes(ax)
for l in d_l:
    ax.plot(bins[:-1], H_mean[:, l], color=c_state_hist[l], zorder=2)
    ax.fill_between(bins[:-1], H_CI_low[:, l], H_CI_high[:, l],
                    facecolor=c_state_fill[l], linewidth=0, zorder=1, alpha=0.25)
#ax.set_xlabel('CS relative afterhyperpolarization (mV)')
#ax.set_ylabel('proportion of CS')
ax.set_xlim([-20, 3])
ax.set_xticks([-20, -10, 0])
ax.set_xlabel('hyperpolarization (mV)')
ax.set_ylim([0, 0.5])
ax.set_yticks([0, 0.25, 0.5])
ax.set_yticklabels([0, '', 0.5])
plt.savefig(os.path.join(fig_folder, ntl[d_l[-1]]+'_CS_ahp.png'), transparent=True)


# do the stats for the above figure
S = np.full([len(data), len(ntl)], np.nan)
for i in np.arange(len(data)):
    for l in np.arange(len(ntl)):
        CS_rel_ahp_Vm = (data[i]['CS_stop_Vm'][data[i][ntl[l]+'_CS_bool']] - 
                        data[i]['CS_start_Vm'][data[i][ntl[l]+'_CS_bool']])
        if CS_rel_ahp_Vm.size > 0:
            S[i, l] = np.nanmedian(CS_rel_ahp_Vm)
# remove extra recordings from cells        
S = S[keep_cells, :]

## do the friedman test (nonparametric repeated measures anova)
## remove cells that have any nans
#S_nonan = S[np.all(~np.isnan(S), axis=1), :]
#X2, p_fried = stats.friedmanchisquare(S_nonan[:, 0], S_nonan[:, 1], S_nonan[:, 2])
#X2, p_fried = stats.friedmanchisquare(S[:, 0], S[:, 1], S[:, 2])

# do the paired boot stats
num_b = 1000
p = np.full(len(ntl) - 1, np.nan)
d = np.full(len(ntl) - 1, np.nan)
for l in np.arange(len(ntl) - 1):
    dif = S[:, l+1] - S[:, 0]
    # remove nans
    dif = dif[~np.isnan(dif)]
    d[l], p[l] = boot_pair_t(dif, num_b)
    print(dif.size)
print(d)
print(p)

# do the paired boot stats for theta hyp and LIA dep cells only
num_b = 1000
# theta hyp cells
dif = S[:, 1][theta_cell_p < 0.05] - S[:, 0][theta_cell_p < 0.05]
# remove nans
dif = dif[~np.isnan(dif)]
d, p = boot_pair_t(dif, num_b)
print(dif.size)
print(d)
print(p)
# LIA dep cells
dif = S[:, 2][LIA_cell_p > 0.95] - S[:, 0][LIA_cell_p > 0.95]
# remove nans
dif = dif[~np.isnan(dif)]
d, p = boot_pair_t(dif, num_b)
print(dif.size)
print(d)
print(p)
# theta nonhyp cells
dif = dif = S[:, 1][theta_cell_p > 0.05] - S[:, 0][theta_cell_p > 0.05]
# remove nans
dif = dif[~np.isnan(dif)]
d, p = boot_pair_t(dif, num_b)
print(dif.size)
print(d)
print(p)

# descriptive numbers
l = 1
np.sum(~np.isnan(S[:, l]))
np.nanmedian(S[:, l])
np.nanstd(S[:, l])
MADAM(S[:, l], np.nanmedian(S[:, l]))




#%% CS-based stats for the average histograms

measure = 'num_sp'
measure = 'CS_dur'
measure = 'CS_height_Vm'
measure = 'CS_rel_ahp_Vm'

num_b = 1000
g0 = CS_ntl[0][measure]
g1 = CS_ntl[1][measure]
g2 = CS_ntl[2][measure]
groups_list = [g0, g1, g2]
real_F, p_boot = boot_anova(groups_list, num_b)
# try the stats test again with a kruskal-wallace (nonparametric 1-way anova)
H, p_kw = stats.kruskal(g0, g1, g2, nan_policy='omit')

# do the pairwise t-tests
boot_t(g0, g1, 1000)
boot_t(g0, g2, 1000)
boot_t(g1, g2, 1000)

# do the 2-sample Kolmogorov–Smirnov test (good for bimodal distributions?)
stats.ks_2samp(g0, g1)
stats.ks_2samp(g0, g2)
stats.ks_2samp(g1, g2)
  
# some numbers from the histogram
l = 1
CS_ntl[l][measure].size
np.nanmedian(CS_ntl[l][measure])
np.nanstd(CS_ntl[l][measure])
MADAM(CS_ntl[l][measure], np.nanmedian(CS_ntl[l][measure]))





# %% make CS rate and index figures

keep_cells = [isinstance(d['cell_id'], int) for d in data]
theta_cell_p = np.array([d['theta_cell_p'] for d in data])[keep_cells]
LIA_cell_p = np.array([d['LIA_cell_p'] for d in data])[keep_cells]


## find which cells have a significant change - boot
#anova_cells = np.full(len(data), np.nan)
#t_boot_cells = np.full([len(data), len(states)], np.nan)
#real_d_cells = np.full([len(data), len(states)], np.nan)
#num_b = 1000
#for i in np.arange(len(data)):
#    groups_list = [data[i]['nost_CS_rate'], data[i]['theta_CS_rate'],
#                   data[i]['LIA_CS_rate']]
#    real_F, anova_cells[i] = boot_anova(groups_list, num_b)
#    # if the anova is significant, do the adhoc stats
#    if anova_cells[i] < 0.05:
#        for l in np.arange(len(states)):
#            real_d_cells[i, l], t_boot_cells[i, l] = boot_t(groups_list[0], groups_list[l+1], num_b)
## remove extra recordings
#anova_cells = anova_cells[keep_cells]
#t_boot_cells = t_boot_cells[keep_cells, :]
#real_d_cells = real_d_cells[keep_cells, :]   

# find which cells have a significant change - nonparametric stats
p_kw = np.full(len(data), np.nan)
p_mw = np.full([len(data), len(states)], np.nan)
num_b = 1000
for i in np.arange(len(data)):
    groups_list = [data[i]['nost_CS_rate'], data[i]['theta_CS_rate'],
                   data[i]['LIA_CS_rate']]
    try:
        H, p_kw[i] = stats.kruskal(groups_list[0], groups_list[1], groups_list[2],
                                   nan_policy='omit')
    except ValueError:
        p_kw[i] = np.nan
    # if the anova is significant, do the adhoc stats
    if p_kw[i] < 0.05:
        for l in np.arange(len(states)):
            U, p_mw[i, l] = stats.mannwhitneyu(groups_list[0], groups_list[l+1],
                                               alternative='two-sided')
# remove extra recordings
p_kw = p_kw[keep_cells]
p_mw = p_mw[keep_cells, :]




# each cells' average frequency of CS during theta, LIA, and no state
# Version: theta and LIA separate
# prep numbers
# only take first recording from each cell
S = np.full([len(data), len(ntl)], np.nan)
for i in np.arange(len(data)):
    for l in np.arange(len(ntl)):
        num_CS = np.sum([data[i][ntl[l]+'_CS_bool']])
        total_time = np.sum(data[i][ntl[l]+'_stop'] - data[i][ntl[l]+'_start'])
        S[i, l] = num_CS/total_time
# remove extra recordings from cells        
S = S[keep_cells, :]        
# plot the stack plot for cell values for each state
fig, ax = plt.subplots(1, figsize=[2.3, 2])
line_x = np.array([1.75, 3.25])
bar_x = np.array([1, 4])
y = S[:, d_l]
for i in np.arange(y.shape[0]):
    ax.plot(line_x, y[i, :], color=c_lgry, zorder=1)
    if d_l == [0, 1]:
        if theta_cell_p[i] < 0.05:
            ax.plot(line_x, y[i, :], color=rgb2hex(c_hyp), zorder=2)
        if theta_cell_p[i] > 0.95:
            ax.plot(line_x, y[i, :], color=rgb2hex(c_dep), zorder=2)
    elif d_l == [0, 2]:
        if LIA_cell_p[i] < 0.05:
            ax.plot(line_x, y[i, :], color=rgb2hex(c_hyp), zorder=2)
        if LIA_cell_p[i] > 0.95:
            ax.plot(line_x, y[i, :], color=rgb2hex(c_dep), zorder=2)
for l in np.arange(y.shape[1]):
    # remove nans
    no_nan = y[:, l]
    no_nan = no_nan[~np.isnan(no_nan)]
    bp = ax.boxplot(no_nan, sym='', patch_artist=True,
                         whis=[5, 95], widths=0.75, positions=[bar_x[l]])     
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=c_state[d_l[l]], linewidth=1.5)
    for patch in bp['boxes']:
        patch.set(facecolor=c_wht)
ax.set_xticks(bar_x)
ax.xaxis.set_tick_params(length=0)
ax.set_xticklabels(['unlabeled', 'theta'])
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
ax.set_yticklabels([0, '', 0.4, '', 0.8])
ax.set_ylabel('Cs rate (Hz)')
ax.set_xlim([0, bar_x[1]+1])
ax.spines['bottom'].set_visible(False)
fig.tight_layout()
plt.savefig(os.path.join(fig_folder, ntl[d_l[-1]]+'_CS_rate.png'), transparent=True)



# do the paired boot stats
num_b = 1000
p = np.full(len(ntl) - 1, np.nan)
d = np.full(len(ntl) - 1, np.nan)
for l in np.arange(len(ntl) - 1):
    dif = S[:, l+1] - S[:, 0]
    # remove nans
    dif = dif[~np.isnan(dif)]
    d[l], p[l] = boot_pair_t(dif, num_b)
print(d)
print(p)

# do the paired boot stats for theta hyp and LIA dep cells only
num_b = 1000
# theta hyp cells
dif = S[:, 1][theta_cell_p < 0.05] - S[:, 0][theta_cell_p < 0.05]
# remove nans
dif = dif[~np.isnan(dif)]
d, p = boot_pair_t(dif, num_b)
print(dif.size)
print(d)
print(p)
# LIA dep cells
dif = S[:, 2][LIA_cell_p > 0.95] - S[:, 0][LIA_cell_p > 0.95]
# remove nans
dif = dif[~np.isnan(dif)]
d, p = boot_pair_t(dif, num_b)
print(dif.size)
print(d)
print(p)


# descriptive numbers
l = 1
np.sum(~np.isnan(S[:, l]))
np.nanmedian(S[:, l])
np.nanstd(S[:, l])
MADAM(S[:, l], np.nanmedian(S[:, l]))



    
    
# find which cells have a significant change - nonparametric stats
p_kw = np.full(len(data), np.nan)
p_mw = np.full([len(data), len(states)], np.nan)
num_b = 1000
for i in np.arange(len(data)):
    groups_list = [data[i]['nost_CS_perc'], data[i]['theta_CS_perc'],
                   data[i]['LIA_CS_perc']]
    # do the kruskall-wallace if not all the CS_perc values are nan
    if ~np.all(np.isnan(np.concatenate(groups_list))):
        try:
            H, p_kw[i] = stats.kruskal(groups_list[0], groups_list[1], groups_list[2],
                                       nan_policy='omit')
        except ValueError:
            p_kw[i] = np.nan
    # if the anova is significant, do the adhoc stats
    if p_kw[i] < 0.05:
        for l in np.arange(len(states)):
            # remove nans before running the test
            g0 = groups_list[0]
            g0 = g0[~np.isnan(g0)]
            g1 = groups_list[l+1]
            g1 = g1[~np.isnan(g1)]
            U, p_mw[i, l] = stats.mannwhitneyu(g0, g1,
                                               alternative='two-sided')
# remove extra recordings
p_kw = p_kw[keep_cells]
p_mw = p_mw[keep_cells, :]


# %% CS index

# each cells' CS index during theta, LIA, and no state
# Version: theta and LIA separate
# prep numbers
# only take first recording from each cell
S = np.full([len(data), len(ntl)], np.nan)
for i in np.arange(len(data)):
    for l in np.arange(len(ntl)):
        num_CS_spikes = np.sum(np.array([c.size for c in data[i]['CS_ind']])[data[i][ntl[l]+'_CS_bool']])
        total_spikes = np.sum([data[i][ntl[l]+'_spike_bool']])
        CS_perc = num_CS_spikes/total_spikes
        if CS_perc > 1:
            CS_perc = 1
        S[i, l] = CS_perc
# remove extra recordings from cells        
S = S[keep_cells, :]        
# plot the stack plot for cell values for each state
fig, ax = plt.subplots(1, figsize=[2.3, 2])
line_x = np.array([1.75, 3.25])
bar_x = np.array([1, 4])
y = S[:, d_l]
for i in np.arange(y.shape[0]):
    ax.plot(line_x, y[i, :], color=c_lgry, zorder=1)
    if d_l == [0, 1]:
        if theta_cell_p[i] < 0.05:
            ax.plot(line_x, y[i, :], color=rgb2hex(c_hyp), zorder=2)
        if theta_cell_p[i] > 0.95:
            ax.plot(line_x, y[i, :], color=rgb2hex(c_dep), zorder=2)
    elif d_l == [0, 2]:
        if LIA_cell_p[i] < 0.05:
            ax.plot(line_x, y[i, :], color=rgb2hex(c_hyp), zorder=2)
        if LIA_cell_p[i] > 0.95:
            ax.plot(line_x, y[i, :], color=rgb2hex(c_dep), zorder=2)
for l in np.arange(y.shape[1]):
    # remove nans
    no_nan = y[:, l]
    no_nan = no_nan[~np.isnan(no_nan)]
    bp = ax.boxplot(no_nan, sym='', patch_artist=True,
                         whis=[5, 95], widths=0.75, positions=[bar_x[l]])      
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=c_state[d_l[l]], linewidth=1.5)
    for patch in bp['boxes']:
        patch.set(facecolor=c_wht)
ax.set_xticks(bar_x)
ax.xaxis.set_tick_params(length=0)
ax.set_xticklabels(['unlabeled', 'theta'])
ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
ax.set_yticklabels([0, '', 0.5, '', 1])
ax.set_ylabel('Cs index')
ax.set_xlim([0, bar_x[1]+1])
ax.spines['bottom'].set_visible(False)
fig.tight_layout()
plt.savefig(os.path.join(fig_folder, ntl[d_l[-1]]+'_CS_index.png'), transparent=True)

# do the paired boot stats
num_b = 1000
p = np.full(len(ntl) - 1, np.nan)
d = np.full(len(ntl) - 1, np.nan)
for l in np.arange(len(ntl) - 1):
    dif = S[:, l+1] - S[:, 0]
    # remove nans
    dif = dif[~np.isnan(dif)]
    d[l], p[l] = boot_pair_t(dif, num_b)
print(d)
print(p)

# do the paired boot stats for theta hyp and LIA dep cells only
num_b = 1000
# theta hyp cells
dif = S[:, 1][theta_cell_p < 0.05] - S[:, 0][theta_cell_p < 0.05]
# remove nans
dif = dif[~np.isnan(dif)]
d, p = boot_pair_t(dif, num_b)
print(dif.size)
print(d)
print(p)
# LIA dep cells
dif = S[:, 2][LIA_cell_p > 0.95] - S[:, 0][LIA_cell_p > 0.95]
# remove nans
dif = dif[~np.isnan(dif)]
d, p = boot_pair_t(dif, num_b)
print(dif.size)
print(d)
print(p)

# descriptive numbers
l = 1
np.sum(~np.isnan(S[:, l]))
np.nanmedian(S[:, l])
np.nanstd(S[:, l])
MADAM(S[:, l], np.nanmedian(S[:, l]))


# %% make figures - dCSI vs dVm


l = 0       
state='theta'
         
# Event-based correlation between dVm and change in CS index
unique_cells = [isinstance(d['cell_id'], int) for d in data]
fig, ax = plt.subplots(1, figsize=[2.25, 2.25])
n = 0
for i in np.arange(len(data)):
    for j in np.arange(data[i][state+'_start'].size):
        x = data[i][state+'_dVm'][j]
        z = data[i][state+'_dVm_p'][j]
        # calculate the CS index in the before window
        CS_bef = np.logical_and(data[i][state+'_CS_spikes'][j] > states[l]['bef'],
                                data[i][state+'_CS_spikes'][j] < states[l]['bef'] + states[l]['samp_time'])
        CS_bef = np.sum(CS_bef)
        nonCS_bef = np.logical_and(data[i][state+'_nonCS_spikes'][j] > states[l]['bef'],
                                   data[i][state+'_nonCS_spikes'][j] < states[l]['bef'] + states[l]['samp_time'])
        nonCS_bef = np.sum(nonCS_bef)
        CSindex_bef = CS_bef/(CS_bef+nonCS_bef)
        # calculate the CS index in the after window
        CS_aft = np.logical_and(data[i][state+'_CS_spikes'][j] > states[l]['aft'],
                                data[i][state+'_CS_spikes'][j] < states[l]['aft'] + states[l]['samp_time'])
        CS_aft = np.sum(CS_aft)
        nonCS_aft = np.logical_and(data[i][state+'_nonCS_spikes'][j] > states[l]['aft'],
                                   data[i][state+'_nonCS_spikes'][j] < states[l]['aft'] + states[l]['samp_time'])
        nonCS_aft = np.sum(nonCS_aft)
        CSindex_aft = CS_aft/(CS_aft+nonCS_aft)
        if np.logical_and(CS_bef+nonCS_bef == 0, CS_aft+nonCS_aft == 0):
            y = np.nan
        else:
            y = CSindex_aft-CSindex_bef
        if np.isnan(y) == False:
            n = n+1
        if z > 0.05:
            ax.scatter(x, y, s=5, facecolors='none', edgecolors=c_mgry, alpha=1, zorder=1)
        elif x < 0:
            ax.scatter(x, y, s=5, facecolors=c_lhyp, edgecolors=c_lhyp, alpha=1, zorder=2)
        elif x > 0:
            ax.scatter(x, y, s=5, facecolors=c_ldep, edgecolors=c_ldep, alpha=1, zorder=2)
ax.axhline(0, linestyle='--', color=c_blk, zorder=1)
ax.axvline(0, linestyle='--', color=c_blk, zorder=1)
ax.set_ylim([-1.1, 1.1])
ax.set_xlim([-18, 18])
# cell-based dVm vs change in CS index
# prep numbers for dVm
all_dVm = np.array([d[state+'_mean_dVm'] for d in data])[[isinstance(d['cell_id'], int) for d in data]]
all_cell_p = np.array([d[state+'_cell_p'] for d in data])[[isinstance(d['cell_id'], int) for d in data]]
keep_cells = np.logical_or(np.isnan(all_dVm), np.isnan(all_cell_p))==0
all_dVm = all_dVm[keep_cells]
all_cell_p = all_cell_p[keep_cells]
cell_hyp_sig = all_dVm[all_cell_p < 0.05]
cell_hyp_no = all_dVm[(all_dVm < 0) & (all_cell_p >= 0.05)]
cell_dep_sig = all_dVm[all_cell_p > 0.95]
cell_dep_no = all_dVm[(all_dVm > 0) & (all_cell_p <= 0.95)]
# prep number for CS index
dCSI = np.full(len(data), np.nan)
for i in np.arange(len(data)):
    dCSI_cell = np.full(data[i][state+'_start'].size, np.nan)
    for j in np.arange(data[i][state+'_start'].size):
        # calculate the CS index in the before window
        CS_bef = np.logical_and(data[i][state+'_CS_spikes'][j] > states[l]['bef'],
                                data[i][state+'_CS_spikes'][j] < states[l]['bef'] + states[l]['samp_time'])
        CS_bef = np.sum(CS_bef)
        nonCS_bef = np.logical_and(data[i][state+'_nonCS_spikes'][j] > states[l]['bef'],
                                   data[i][state+'_nonCS_spikes'][j] < states[l]['bef'] + states[l]['samp_time'])
        nonCS_bef = np.sum(nonCS_bef)
        CSindex_bef = CS_bef/(CS_bef+nonCS_bef)
        # calculate the CS index in the after window
        CS_aft = np.logical_and(data[i][state+'_CS_spikes'][j] > states[l]['aft'],
                                data[i][state+'_CS_spikes'][j] < states[l]['aft'] + states[l]['samp_time'])
        CS_aft = np.sum(CS_aft)
        nonCS_aft = np.logical_and(data[i][state+'_nonCS_spikes'][j] > states[l]['aft'],
                                   data[i][state+'_nonCS_spikes'][j] < states[l]['aft'] + states[l]['samp_time'])
        nonCS_aft = np.sum(nonCS_aft)
        CSindex_aft = CS_aft/(CS_aft+nonCS_aft)
        if np.logical_and(CS_bef+nonCS_bef == 0, CS_aft+nonCS_aft == 0):
            dCSI_cell[j] = np.nan
        else:
            dCSI_cell[j] = CSindex_aft-CSindex_bef
    dCSI[i] = np.nanmean(dCSI_cell)
dCSI = dCSI[unique_cells]
dCSI = dCSI[keep_cells]
cell_hyp_sig_dCSI = dCSI[all_cell_p < 0.05]
cell_hyp_no_dCSI = dCSI[(all_dVm < 0) & (all_cell_p >= 0.05)]
cell_dep_sig_dCSI = dCSI[all_cell_p > 0.95]
cell_dep_no_dCSI = dCSI[(all_dVm > 0) & (all_cell_p <= 0.95)]         
# add the cell dots on top
s_cell = 20
ax.scatter(cell_hyp_sig, cell_hyp_sig_dCSI, s=s_cell, facecolors=c_hyp,
           edgecolors=c_blk, zorder=3, alpha=1)
ax.scatter(cell_hyp_no, cell_hyp_no_dCSI, s=s_cell, facecolors='none',
           edgecolors=c_blk, zorder=3, alpha=1)
ax.scatter(cell_dep_sig, cell_dep_sig_dCSI, s=s_cell, facecolors=rgb2hex(c_dep),
           edgecolors=rgb2hex(c_blk), zorder=3, alpha=1)
ax.scatter(cell_dep_no, cell_dep_no_dCSI, s=s_cell, facecolors='none',
           edgecolors=c_blk, zorder=3, alpha=1)
ax.set_xlabel(r'$\Delta$'+' Vm (mV)')
ax.set_ylabel(r'$\Delta$'+' CS index')
fig.tight_layout()
plt.savefig(os.path.join(fig_folder, 'dCSI_vs_dVm_'+state+'.png'), transparent=True)
