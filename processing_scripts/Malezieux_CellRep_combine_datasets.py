# -*- coding: utf-8 -*-
"""
Created on Mon May 25 09:49:29 2020

@author: ashley
"""

# Manuscript Malezieux, Kees, Mulle submitted to Cell Reports
# combine all datasets (except Rin) into one dataset for upload to repository


# %% import modules

import os
import numpy as np
from scipy import stats
from scipy import signal
import pandas as pd


#%% definitions

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


# %% Load data
    
dataset_folder = (r'C:\Users\akees\Documents\Ashley\Analysis\MIND' +
                  r'\Python\Datasets\MIND-1\MIND1_v3\noSxx')

cell_files = os.listdir(dataset_folder)
data = [{} for k in np.arange(len(cell_files))]
for i in np.arange(len(cell_files)):
    full_file = os.path.join(dataset_folder, cell_files[i])
    data[i] = np.load(full_file, allow_pickle=True).item()


## Load complex spike data
# NOTE: sp_times will be replaced by the sp_times created by Manu_MIND1_load_data_CS

CS_folder = (r'C:\Users\akees\Documents\Ashley\Analysis\MIND' +
                  r'\Python\Datasets\MIND-1\MIND1_v3\CS')
   
CS_files = os.listdir(CS_folder)
for i in np.arange(len(CS_files)):
    full_file = os.path.join(CS_folder, CS_files[i])
    CS_data = np.load(full_file, allow_pickle=True).item()
    data[i].update(CS_data)


## Load the hf LFP and Vm traces

hf_folder = (r'C:\Users\akees\Documents\Ashley\Analysis\MIND' +
                  r'\Python\Datasets\MIND-1\MIND1_v3\hf')
   
hf_files = os.listdir(hf_folder)
for i in np.arange(len(hf_files)):
    full_file = os.path.join(hf_folder, hf_files[i])
    hf_data = np.load(full_file, allow_pickle=True).item()
    data[i].update(hf_data)
    

# %% 
    
states = [{'id':'theta', 'bef':-2.5, 'aft':0.5, 'samp_time':2, 't_win':[-3, 3]},
          {'id':'LIA', 'bef':-4, 'aft':-1, 'samp_time':2, 't_win':[-4, 2]}]



#%% analysis across continuous time

# goal = dataframe with measures for each window

# note on windows:
# In Welch's method for making the psd, it's good to have a hann window (equivalent
# to a tukey window with taper fraction = 1) with 50% overlap - because the psds 
# will eventually be averaged
# In a spectrogram, it's good to keep the windows as independent as possible,
# so the suggestion is widening the hann window to a tukey with taper fraction = 0.25
# and an overlap of 1/8 of the window
# Problem = signal.coherence needs at least 2 segments in the window, so make
# windows twice the length of the segment, change the window to hann, and overlap 25%
nperseg = 1024  # segments to get good frequency resolution
noverlap = nperseg/2  # 50% overlap in segments to calculate psd/csd/coherence
win_len = 2*nperseg  # window can accomodate at least 2 segments
win_overlap = win_len/4  # windows overlap 25%
theta_low = 6
theta_high = 12
for i in np.arange(len(data)):
    coh_df = pd.DataFrame()
    fs = 1/np.round(data[i]['hf_ts'][1] - data[i]['hf_ts'][0], 5)
    # find the left edges of the windows
    win = np.arange(0, data[i]['hf_ts'].size, win_len-win_overlap, dtype=int)
    if win[-1]+win_len > data[i]['hf_ts'].size:
        win = win[:-1]
    # make the timestamp vector
    ts = data[i]['hf_ts'][(win+win_len/2).astype(int)]
    # make calculations over each window
    Vm_mode = np.full(win.size, np.nan)
    power_Vm = np.full(win.size, np.nan)
    power_lfp = np.full(win.size, np.nan)
    peak_f_Vm = np.full(win.size, np.nan)
    peak_f_lfp = np.full(win.size, np.nan)
    coh_Vm = np.full(win.size, np.nan)
    coh_lfp = np.full(win.size, np.nan)
    ph_Vm = np.full(win.size, np.nan)
    ph_lfp = np.full(win.size, np.nan)
    for t in np.arange(ts.size):
        # take the segment of Vm and lfp
        Vm = data[i]['Vm_hf'][win[t]:win[t]+win_len]
        lfp = data[i]['lfp_hf'][win[t]:win[t]+win_len]
        Vm_nosp = data[i]['Vm_hf_nosp'][win[t]:win[t]+win_len]
        # find the mode of the Vm during the window
        mode, count = stats.mode(np.round(Vm_nosp, decimals=1), nan_policy='omit')
        Vm_mode[t] = mode[0]
        # calculate the psd for lfp and Vm
        f, Pxx_Vm = signal.welch(Vm, fs=fs, window=('tukey', 1), nperseg=nperseg,
                                 noverlap=noverlap, scaling='spectrum')
        f, Pxx_lfp = signal.welch(lfp, fs=fs, window=('tukey', 1), nperseg=nperseg,
                                  noverlap=noverlap, scaling='spectrum')
        f, Cxy = signal.coherence(Vm, lfp, fs=fs, window=('tukey', 1), nperseg=nperseg,
                                  noverlap=noverlap)
        f, Pxy = signal.csd(Vm, lfp, fs=fs,  window=('tukey', 1), nperseg=nperseg,
                                  noverlap=noverlap)
        Pxy = np.angle(Pxy, deg=True)
        # detect the peak frequency in the theta band for Vm and lfp
        f_ind0 = np.searchsorted(f, theta_low)
        f_ind1 = np.searchsorted(f, theta_high)
        # for Vm
        Vm_ind = np.argmax(Pxx_Vm[f_ind0:f_ind1])+f_ind0
        peak_f_Vm[t] = f[Vm_ind]
        power_Vm[t] = Pxx_Vm[Vm_ind]
        coh_Vm[t] = Cxy[Vm_ind]
        ph_Vm[t] = Pxy[Vm_ind]
        # for LFP
        lfp_ind = np.argmax(Pxx_lfp[f_ind0:f_ind1])+f_ind0
        peak_f_lfp[t] = f[lfp_ind]
        power_lfp[t] = Pxx_lfp[lfp_ind]
        coh_lfp[t] = Cxy[lfp_ind]
        ph_lfp[t] = Pxy[lfp_ind]
    # save values as separate numpy arrays
    data[i]['Vm_mode'] = Vm_mode
    data[i]['coh_ts'] = ts
    data[i]['peak_f_Vm'] = peak_f_Vm
    data[i]['peak_f_lfp'] = peak_f_lfp
    data[i]['power_Vm'] = power_Vm
    data[i]['power_lfp'] = power_lfp
    data[i]['coh_Vm'] = coh_Vm
    data[i]['coh_lfp'] = coh_lfp
    data[i]['ph_Vm'] = ph_Vm
    data[i]['ph_lfp'] = ph_lfp




# %% for each event, find the Vm0, dVm and p-value of the event dVm

# find Vm0, dVm and significance for each run, excluding when Ih is changed
for l in np.arange(len(states)):
    for i in np.arange(len(data)):
        samp_freq = 1/(data[i]['Vm_ds_ts'][1] - data[i]['Vm_ds_ts'][0])
        num_ind = int(states[l]['samp_time']*samp_freq)
        # find index of dIh_times
        dIh_ind = data[i]['dIh_times']*samp_freq
        dIh_ind = dIh_ind.astype(int)
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
                Vm0[j] = np.nan
                dVm[j] = np.nan
                dVm_p[j] = np.nan
            else:
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
        data[i][states[l]['id']+'_Vm0'] = Vm0
        data[i][states[l]['id']+'_dVm'] = dVm
        data[i][states[l]['id']+'_dVm_p'] = dVm_p


# %% add event-based data to coherence timeseries dataframes
        
for i in np.arange(len(data)):
    coh_states = np.full(data[i]['coh_ts'].size, 'nost', dtype='object')
    coh_run_theta = np.zeros(data[i]['coh_ts'].size, dtype='bool')
    Vm0 = np.full(data[i]['coh_ts'].size, np.nan)
    dVm = np.full(data[i]['coh_ts'].size, np.nan)
    dVm_p = np.full(data[i]['coh_ts'].size, np.nan)
    # do LIA then theta, in case there are any windows that operlap
    # i.e. give priority to the theta values
    for l in [1, 0]:
        for j in np.arange(data[i][states[l]['id']+'_start'].size):
            ind0 = np.searchsorted(data[i]['coh_ts'],
                                   data[i][states[l]['id']+'_start'][j])
            ind1 = np.searchsorted(data[i]['coh_ts'],
                                   data[i][states[l]['id']+'_stop'][j])
            coh_states[ind0:ind1] = states[l]['id']
            Vm0[ind0:ind1] = data[i][states[l]['id']+'_Vm0'][j]
            dVm[ind0:ind1] = data[i][states[l]['id']+'_dVm'][j]
            dVm_p[ind0:ind1] = data[i][states[l]['id']+'_dVm_p'][j]
            # cross check whether events match those in run_theta
            if np.any(np.isin(data[i]['run_theta_start'], data[i][states[l]['id']+'_start'][j])):
                coh_run_theta[ind0:ind1] = True      
    # add the Ih
    inds = np.searchsorted(data[i]['Vm_Ih_ts'], data[i]['coh_ts'])
    # if the last ind is the final ind, subtract 1
    if inds[-1] == data[i]['Vm_Ih_ts'].size:
        inds[-1] = inds[-1]-1
    Ih = data[i]['Vm_Ih'][inds]
    # put nans at the timestamp before and after the Ih change
    for h in np.arange(data[i]['dIh_times'].size):
        ind0 = np.searchsorted(data[i]['coh_ts'], data[i]['dIh_times'][h])
        Ih[ind0] = np.nan
        Ih[ind0-1] = np.nan
    data[i]['coh_states'] = coh_states
    data[i]['coh_run_theta'] = coh_run_theta
    data[i]['coh_Ih'] = Ih
    data[i]['coh_Vm0'] = Vm0
    data[i]['coh_dVm'] = dVm
    data[i]['coh_dVm_p'] = dVm_p
 
    
# make the cell dataframes and concatenate into one big dataframe
for i in np.arange(len(data)):
    coh_df = pd.DataFrame()
    coh_df['Vm'] = data[i]['Vm_mode']
    coh_df['ts'] = data[i]['coh_ts']
    coh_df['peak_f_Vm'] = data[i]['peak_f_Vm']
    coh_df['peak_f_lfp'] = data[i]['peak_f_lfp']
    coh_df['power_Vm'] = data[i]['power_Vm']
    coh_df['z_power_Vm'] = stats.zscore(data[i]['power_Vm'])
    coh_df['power_lfp'] = data[i]['power_lfp']
    coh_df['coh_Vm'] = data[i]['coh_Vm']
    coh_df['coh_lfp'] = data[i]['coh_lfp']
    coh_df['ph_Vm'] = data[i]['ph_Vm']
    coh_df['ph_lfp'] = data[i]['ph_lfp']
    coh_df['state'] = data[i]['coh_states']
    coh_df['run_theta'] = data[i]['coh_run_theta']
    coh_df['Ih'] = data[i]['coh_Ih']
    coh_df['Vm0'] = data[i]['coh_Vm0']
    coh_df['dVm'] = data[i]['coh_dVm']
    coh_df['dVm_p'] = data[i]['coh_dVm_p']
    # add the cell id (include multiple recordings, but include under same label)
    # record the cell_p for theta and LIA from the first recording from the cell
    if isinstance(data[i]['cell_id'], str):
        ind = data[i]['cell_id'].find('_')
        cell_int = int(data[i]['cell_id'][:ind])
        coh_df['cell_id'] = np.full(data[i]['coh_ts'].size, cell_int)
        cell_ind = int(np.where(np.array([d['cell_id'] for d in data]) == str(cell_int))[0])
        coh_df['theta_cell_p'] = np.full(data[i]['coh_ts'].size, data[cell_ind]['theta_cell_p'])
        coh_df['LIA_cell_p'] = np.full(data[i]['coh_ts'].size, data[cell_ind]['LIA_cell_p'])
    else:
        cell_int = data[i]['cell_id']
        coh_df['cell_id'] = np.full(data[i]['coh_ts'].size, cell_int)
        coh_df['theta_cell_p'] = np.full(data[i]['coh_ts'].size, data[i]['theta_cell_p'])
        coh_df['LIA_cell_p'] = np.full(data[i]['coh_ts'].size, data[i]['LIA_cell_p']) 
    data[i]['coh_df'] = coh_df





#%% process CS
    
# for each CS, find the relevant values in the suthreshold Vm trace:
# Vm at CS start, maximum Vm, and Vm at CS end (hyperpolarization)
for i in np.arange(len(data)):
    CS_start_Vm = data[i]['Vm_sub'][np.searchsorted(data[i]['Vm_sub_ts'], data[i]['CS_start'])]
    CS_stop_Vm = data[i]['Vm_sub'][np.searchsorted(data[i]['Vm_sub_ts'], data[i]['CS_stop'])]
    CS_max_Vm = np.full(data[i]['CS_start'].size, np.nan)
    for j in np.arange(data[i]['CS_start'].size):
        ind1 = np.searchsorted(data[i]['Vm_sub_ts'], data[i]['CS_start'][j])
        ind2 = np.searchsorted(data[i]['Vm_sub_ts'], data[i]['CS_stop'][j])
        CS_max_Vm[j] = np.nanmax(data[i]['Vm_sub'][ind1:ind2])
    data[i]['CS_start_Vm'] = CS_start_Vm
    data[i]['CS_stop_Vm'] = CS_stop_Vm
    data[i]['CS_max_Vm'] = CS_max_Vm


# %% process spikelets
    
# for each cell, find start and stop times for unlabeled times
for i in np.arange(len(data)):
    state_start = np.concatenate([data[i]['theta_start'], data[i]['LIA_start']])
    state_start = np.sort(state_start)
    state_stop = np.concatenate([data[i]['theta_stop'], data[i]['LIA_stop']])
    state_stop = np.sort(state_stop)
    data[i]['nost_start'] = np.append(data[i]['Vm_ds_ts'][0], state_stop)
    data[i]['nost_stop'] = np.append(state_start, data[i]['Vm_ds_ts'][-1])

#for each cell, find spikelets times 
for i in np.arange(len(data)):
    data[i]['spikelet_times'] = data[i]['sp_times'][data[i]['spikelets_ind'].astype(int)] 

# for each cell, calculate the ispikeleti (inter-spikelet-interval)
# for spikelets only
for i in np.arange(len(data)):
    if data[i]['spikelet_times'].size > 0:
        isi0 = data[i]['spikelet_times'][0] - data[i]['Vm_ds_ts'][0]
        data[i]['ispikeleti'] = np.ediff1d(data[i]['spikelet_times'], to_begin=isi0)
    else:
        data[i]['ispikeleti'] = np.empty(0)    
    
# for each spikelet, determine which state it occurs in (and those in no state)
for i in np.arange(len(data)):
    nost_sp = np.ones(data[i]['spikelet_times'].size, dtype=bool)
    for l in np.arange(len(states)):
        state_sp = np.zeros(data[i]['spikelet_times'].size, dtype=bool)
        for j in np.arange(data[i][states[l]['id']+'_start'].size):
            # find the spikes that occur in that event
            temp_bool = np.all((data[i]['spikelet_times'] > data[i][states[l]['id']+'_start'][j], 
                                data[i]['spikelet_times'] < data[i][states[l]['id']+'_stop'][j]),
                               axis=0)
            state_sp = state_sp + temp_bool
        data[i][states[l]['id']+'_spikelet_bool'] = np.squeeze(state_sp)
        nost_sp = nost_sp*[state_sp == False]
    data[i]['nost_spikelet_bool'] = np.squeeze(nost_sp)

# for each spikelet during theta, determine if it happens during run or nonrun
for i in np.arange(len(data)):    
    state_sp = np.zeros(data[i]['spikelet_times'].size, dtype=bool)
    for j in np.arange(data[i]['run_theta_start'].size):
        # find the spikes that occur in that event
        temp_bool = np.all((data[i]['spikelet_times'] > data[i]['run_theta_start'][j], 
                            data[i]['spikelet_times'] < data[i]['run_theta_stop'][j]),
                           axis=0)
        state_sp = state_sp + temp_bool
    data[i]['run_theta_spikelet_bool'] = np.squeeze(state_sp)
for i in np.arange(len(data)):        
    state_sp = np.zeros(data[i]['spikelet_times'].size, dtype=bool)
    for l in np.arange(data[i]['nonrun_theta_start'].size):
        # find the spikes that occur in that event
        temp_bool = np.all((data[i]['spikelet_times'] > data[i]['nonrun_theta_start'][l], 
                            data[i]['spikelet_times'] < data[i]['nonrun_theta_stop'][l]),
                           axis=0)
        state_sp = state_sp + temp_bool
    data[i]['nonrun_theta_spikelet_bool'] = np.squeeze(state_sp)


# %% process data - spikes

keep_cells_thresh = np.where([isinstance(d['cell_id'], int) for d in data])[0]

# for each cell, make a new spike_times for specifically non-spikelets
for i in np.arange(len(data)):
    data[i]['spike_times'] = np.delete(data[i]['sp_times'],
                                       data[i]['spikelets_ind'])    

# for each cell, calculate the isi (inter-spike-interval)
# for true spikes only
for i in np.arange(len(data)):
    if data[i]['spike_times'].size > 0:
        isi0 = data[i]['spike_times'][0] - data[i]['Vm_ds_ts'][0]
        data[i]['isi'] = np.ediff1d(data[i]['sp_times'], to_begin=isi0)
    else:
        data[i]['isi'] = np.empty(0)    


# for each cell, identify the spikes than can be used for threshold analysis
# prior isi has to be more than a threshold
# cannot be a following spike in a doublet or CS
# cannot be a spikelet
# must have a peak Vm that passes a threshold
# must have an amplitude that passes a threshold
isi_thresh = 0.05 #0.05  #50 ms
peak_thresh = -10  # mV, absolute Vm
amp_thresh = 0 #35  # mV, spike amplitude
for i in np.arange(len(data)):
    if data[i]['isi'].size > 0:
        isi_ind = np.where(data[i]['isi'] > isi_thresh)[0]
        valid_ind = np.append(data[i]['singles_ind'], data[i]['doublets_ind'][0])
        valid_ind = np.append(valid_ind, np.array([d[0] for d in data[i]['CS_ind']], dtype=int))
        valid_ind = np.sort(valid_ind)
        valid_ind = np.intersect1d(valid_ind, isi_ind)
        peak_ind = np.where(data[i]['sp_peak_Vm'] > peak_thresh)
        amp_ind = np.where(data[i]['sp_peak_Vm'] - data[i]['sp_thresh_Vm'] > amp_thresh)
        peak_ind = np.intersect1d(peak_ind, amp_ind)
        data[i]['th_ind'] = np.intersect1d(valid_ind, peak_ind)
    else:
        data[i]['th_ind'] = np.empty(0, dtype=int)


# save the thresh times and Vm for each cell
for i in np.arange(len(data)):
    if data[i]['sp_times'].size > 0:
        data[i]['thresh_times'] = data[i]['sp_times'][data[i]['th_ind']]
        data[i]['thresh_Vm'] = data[i]['sp_thresh_Vm'][data[i]['th_ind']]
    else:
        data[i]['thresh_times'] = np.empty(0)
        data[i]['thresh_Vm'] = np.empty(0)


# for each spike used in threshold analysis,
# determine which state it occurs in (and those in no state)
for i in np.arange(len(data)):
    nost_sp = np.ones(data[i]['thresh_times'].size, dtype=bool)
    for l in np.arange(len(states)):
        state_sp = np.zeros(data[i]['thresh_times'].size, dtype=bool)
        for j in np.arange(data[i][states[l]['id']+'_start'].size):
            # find the spikes that occur in that event
            temp_bool = np.all((data[i]['thresh_times'] > data[i][states[l]['id']+'_start'][j], 
                                data[i]['thresh_times'] < data[i][states[l]['id']+'_stop'][j]),
                               axis=0)
            state_sp = state_sp + temp_bool
        data[i][states[l]['id']+'_thresh_bool'] = np.squeeze(state_sp)
        nost_sp = nost_sp*[state_sp == False]
    data[i]['nost_thresh_bool'] = np.squeeze(nost_sp)

# for each spike for threshold analysis, determine if it happens during run or nonrun
for i in np.arange(len(data)):    
    state_sp = np.zeros(data[i]['thresh_times'].size, dtype=bool)
    for j in np.arange(data[i]['run_theta_start'].size):
        # find the spikes that occur in that event
        temp_bool = np.all((data[i]['thresh_times'] > data[i]['run_theta_start'][j], 
                            data[i]['thresh_times'] < data[i]['run_theta_stop'][j]),
                           axis=0)
        state_sp = state_sp + temp_bool
    data[i]['run_theta_thresh_bool'] = np.squeeze(state_sp)
for i in np.arange(len(data)):        
    state_sp = np.zeros(data[i]['thresh_times'].size, dtype=bool)
    for l in np.arange(data[i]['nonrun_theta_start'].size):
        # find the spikes that occur in that event
        temp_bool = np.all((data[i]['thresh_times'] > data[i]['nonrun_theta_start'][l], 
                            data[i]['thresh_times'] < data[i]['nonrun_theta_stop'][l]),
                           axis=0)
        state_sp = state_sp + temp_bool
    data[i]['nonrun_theta_thresh_bool'] = np.squeeze(state_sp)


# modify the Vm_sub to cut out the subthreshold components of spikes
search_period = 800
for i in np.arange(len(data)):
    data[i]['Vm_nosubsp'] = remove_spikes_sub(data[i]['Vm_sub_ts'],
                                             data[i]['Vm_sub'],
                                             data[i]['sp_times'], search_period)


# take Vm_sub and Vm_var triggered by spikes used in threshold analysis
t_win = [-1, 0.1]
for i in np.arange(len(data)):
    sp_t_Vm, sp_t_ts = prepare_eta(data[i]['Vm_nosubsp'], data[i]['Vm_sub_ts'],
                                        data[i]['thresh_times'], t_win)
    sp_t_Vm_var, sp_t_ts_var = prepare_eta(data[i]['Vm_var'], data[i]['Vm_ds_ts'],
                                        data[i]['thresh_times'], t_win)
    data[i]['sp_t_Vm'] = sp_t_Vm
    data[i]['sp_t_Vm_var'] = sp_t_Vm_var
sp_t_ts = 1000*sp_t_ts
sp_t_ind0 = np.searchsorted(sp_t_ts, 0)-1
sp_t_ts_var = 1000*sp_t_ts_var


# for each spike, find the relative threshold, and tight and broad variances
# tight variance: variance of the Vm_nosubsp in the window before the spike
# broad variance: mean Vm_var in the window before the spike (this is variance
# over rolling windows of 1 second, and then smoothed over 2 seconds)
psd_win = [-21, -19]; PSD_win = '20'
psd_win = (sp_t_ts > psd_win[0]) & (sp_t_ts < psd_win[1])
p_sp_win = [-300, -50]; Vm_win = '300-50'
#p_sp_win = [-500, -50]; Vm_win = '500-50'
p_sp_win = (sp_t_ts > p_sp_win[0]) & (sp_t_ts < p_sp_win[1])
p_sp_win_var = [-300, -50]; var_win = '300-50'
#p_sp_win_var = [-1000, -50]; var_win = '1000-50'
p_sp_win_tight = (sp_t_ts > p_sp_win_var[0]) & (sp_t_ts < p_sp_win_var[1])
p_sp_win_broad = (sp_t_ts_var > p_sp_win_var[0]) & (sp_t_ts_var < p_sp_win_var[1])
for i in np.arange(len(data)):
    if data[i]['thresh_times'].size > 0:
        psd = np.nanmean(data[i]['sp_t_Vm'][psd_win, :], axis=0) - data[i]['thresh_Vm']
        #th_dist = np.nanmean(data[i]['sp_t_Vm'][p_sp_win, :], axis=0) - data[i]['thresh_Vm']
        #th_dist = np.nanmedian(data[i]['sp_t_Vm'][p_sp_win, :], axis=0) - data[i]['thresh_Vm']
        mode, count = stats.mode(np.round(data[i]['sp_t_Vm'][p_sp_win, :], decimals=1), axis=0, nan_policy='omit')
        th_dist = mode.data - data[i]['thresh_Vm']
        data[i]['psd'] = psd
        data[i]['th_dist'] = np.squeeze(th_dist)
        data[i]['pre_spike_Vm'] = np.squeeze(data[i]['thresh_Vm'] + th_dist)
        data[i]['th_var_tight'] = np.nanvar(data[i]['sp_t_Vm'][p_sp_win_tight, :], axis=0)
        data[i]['th_var_broad'] = np.nanmean(data[i]['sp_t_Vm_var'][p_sp_win_broad, :], axis=0)
    else:
        data[i]['psd'] = np.empty(0)
        data[i]['th_dist'] = np.empty(0)
        data[i]['pre_spike_Vm'] = np.empty(0)
        data[i]['th_var_tight'] = np.empty(0)
        data[i]['th_var_broad'] = np.empty(0)


#%% cell-by-cell comparisons of spikelets

# for each cell, make a dataframe of spikelet properties
for i in np.arange(len(data)):
    spikelet_df = pd.DataFrame()
    if data[i]['spikelets_ind'].size > 0:
        spikelet_df['times'] = data[i]['spikelet_times']
        spikelet_df['max_rise'] = data[i]['sp_max_rise'][data[i]['spikelets_ind']]
        spikelet_df['thresh_Vm'] = data[i]['sp_thresh_Vm'][data[i]['spikelets_ind']]
        spikelet_df['fwhm'] = 1000*data[i]['sp_fwhm'][data[i]['spikelets_ind']]
        spikelet_df['peak_Vm'] = data[i]['sp_peak_Vm'][data[i]['spikelets_ind']]
        spikelet_df['amplitude'] = spikelet_df['peak_Vm'] - spikelet_df['thresh_Vm']
        spikelet_df['rise_time'] = 1000*data[i]['sp_rise_time'][data[i]['spikelets_ind']]
        spikelet_df['decay_tau'] = 1000*data[i]['sp_decay_tau1'][data[i]['spikelets_ind']]
        #spikelet_df['decay_tau2'] = 1000*data[i]['sp_decay_tau2'][data[i]['spikelets_ind']]
        spikelet_df['decay_error'] = data[i]['sp_decay_error'][data[i]['spikelets_ind']]
        # record the state of each spikelet
        spikelet_state = np.full(data[i]['spikelets_ind'].size, 'nost', dtype='object')
        for l in np.arange(len(states)):
            spikelet_state[data[i][states[l]['id']+'_spikelet_bool']] = states[l]['id']
        spikelet_df['state'] = spikelet_state
        # determine whether spikelets are during run or nonrun theta
        spikelet_df['run_theta'] = data[i]['run_theta_spikelet_bool']
        # add the cell id (include multiple recordings, but include under same label)
        if isinstance(data[i]['cell_id'], str):
            ind = data[i]['cell_id'].find('_')
            cell_int = int(data[i]['cell_id'][:ind])
            spikelet_df['cell_id'] = np.full(data[i]['spikelets_ind'].size, cell_int)
        else:
            cell_int = data[i]['cell_id']
            spikelet_df['cell_id'] = np.full(data[i]['spikelets_ind'].size, cell_int)
    data[i]['spikelet_df'] = spikelet_df
    


# %% for each cell, make a dataframe of spike properties


for i in np.arange(len(data)):
    th_df = pd.DataFrame()
    if data[i]['th_ind'].size > 0:
        th_df['times'] = data[i]['thresh_times']
        th_df['pre_sp_Vm'] = data[i]['th_dist'] + data[i]['thresh_Vm']
        th_df['max_rise'] = data[i]['sp_max_rise'][data[i]['th_ind']]
        th_df['thresh_Vm'] = data[i]['sp_thresh_Vm'][data[i]['th_ind']]
        th_df['fwhm'] = 1000*data[i]['sp_fwhm'][data[i]['th_ind']]
        th_df['peak_Vm'] = data[i]['sp_peak_Vm'][data[i]['th_ind']]
        th_df['amplitude'] = th_df['peak_Vm'] - th_df['thresh_Vm']
        th_df['rise_time'] = 1000*data[i]['sp_rise_time'][data[i]['th_ind']]
        th_df['decay_tau'] = 1000*data[i]['sp_decay_tau1'][data[i]['th_ind']]
        #spike_df['decay_tau2'] = 1000*data[i]['sp_decay_tau2'][data[i]['th_ind']]
        th_df['decay_error'] = data[i]['sp_decay_error'][data[i]['th_ind']]
        # record the state of each spikelet
        spike_state = np.full(data[i]['th_ind'].size, 'nost', dtype='object')
        for l in np.arange(len(states)):
            spike_state[data[i][states[l]['id']+'_thresh_bool']] = states[l]['id']
        th_df['state'] = spike_state
        # determine whether spikelets are during run or nonrun theta
        th_df['run_theta'] = data[i]['run_theta_thresh_bool']
        # add the cell id (include multiple recordings, but include under same label)
        if isinstance(data[i]['cell_id'], str):
            ind = data[i]['cell_id'].find('_')
            cell_int = int(data[i]['cell_id'][:ind])
            th_df['cell_id'] = np.full(data[i]['th_ind'].size, cell_int)
        else:
            cell_int = data[i]['cell_id']
            th_df['cell_id'] = np.full(data[i]['th_ind'].size, cell_int)
    data[i]['th_df'] = th_df
    


# %% delete unused keys

keys2delete = ['dial_start', 'dial_stop', 'no_state_dial_start',
               'no_state_dial_stop', 'LIA_dial_start', 'LIA_dial_stop',
               'run_theta_dial_start', 'run_theta_dial_stop',
               'nonrun_theta_dial_start', 'nonrun_theta_dial_stop', 'synch',
               'Vm_hf', 'Vm_hf_nosp', 'hf_ts', 'lfp_hf', 'Vm_sub', 'Vm_sub_ts',
               'Vm_nosubsp', 'Vm_mode', 'coh_ts', 'peak_f_Vm', 'peak_f_lfp',
               'power_Vm', 'power_lfp', 'coh_Vm', 'coh_lfp', 'ph_Vm', 'ph_lfp',
               'theta_Vm0', 'theta_dVm', 'theta_dVm_p', 'LIA_Vm0', 'LIA_dVm',
               'LIA_dVm_p', 'coh_states', 'coh_run_theta', 'coh_Ih', 'coh_Vm0',
               'coh_dVm', 'coh_dVm_p', 'spikelet_times', 'ispikeleti',
               'theta_spikelet_bool', 'LIA_spikelet_bool', 'nost_spikelet_bool',
               'run_theta_spikelet_bool', 'nonrun_theta_spikelet_bool',
               'theta_thresh_bool', 'LIA_thresh_bool', 'nost_thresh_bool',
               'run_theta_thresh_bool', 'nonrun_theta_thresh_bool', 'sp_t_Vm',
               'sp_t_Vm_var', 'th_var_tight',
               'th_var_broad']

for i in np.arange(len(data)):
    for j in np.arange(len(keys2delete)):
        if keys2delete[j] in data[i].keys():
            del data[i][keys2delete[j]]


# %% save dictionaries using numpy   
    
    
for i in np.arange(len(data)):
    np.save((r'C:\Users\akees\Documents\Ashley\Papers\MIND 1\Cell Reports\Dryad upload\Dataset\cell_' + 
             str(data[i]['cell_id']) + '.npy'), data[i])   
    


