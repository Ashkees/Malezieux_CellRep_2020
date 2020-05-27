# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 15:31:32 2019

@author: Ashley
"""

# Manuscript Malezieux, Kees, Mulle submitted to Cell Reports
# Figure 3
# Description: onset and offset kinetics of dVm with theta



# %% import modules

import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from itertools import compress
import matplotlib as mpl


# %% definitions

# eta = event triggered averages.  CHANGE: nans instead of removing events
# VERSION: sample window is more likely to be the same in different recordings
def prepare_eta(signal, ts, event_times, win):
    samp_period = np.round((ts[1] - ts[0]), decimals=3)
    win_npts = [np.round(np.abs(win[0])/samp_period).astype(int),
                np.round(np.abs(win[1])/samp_period).astype(int)]
#    win_npts = [ts[ts < ts[0] + np.abs(win[0])].size,
#                ts[ts < ts[0] + np.abs(win[1])].size]
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

# eta = event triggered averages.
# VERSION: sample window is more likely to be the same in different recordings
# VERSION: Keep good part of signal, put nans when recording ends within the window
    # only remove events where there is no recording for the dVm comparison windows
# Note: only for 1d signals
def prepare_eta_keep(signal, ts, event_times, win, dVm_win):
    samp_period = np.round((ts[1] - ts[0]), decimals=3)
    win_npts = [np.round(np.abs(win[0])/samp_period).astype(int),
                np.round(np.abs(win[1])/samp_period).astype(int)]
    #et_ts = ts[0:np.sum(win_npts)] - ts[0] + win[0]
    et_ts = np.arange(win[0], win[1], samp_period)
    et_signal = np.empty(0)
    # pad signal and ts with nans at front and end
    signal = np.concatenate((np.full(win_npts[0], np.nan), signal,
                             np.full(win_npts[1], np.nan)), axis=None)
    ts_pad = np.concatenate((et_ts[:win_npts[0]]+ts[0], ts,
                             et_ts[win_npts[0]:]+ts[-1]), axis=None)
    if event_times.size > 0:
        et_signal = np.zeros((et_ts.size, event_times.size))
        for i in np.arange(event_times.size):
            if np.logical_or((event_times[i]+dVm_win[0]<ts[0]), (event_times[i]+dVm_win[1]>ts[-1])):
                et_signal[:, i] = np.nan*np.ones(et_ts.size)
            else:
                #ind = np.argmin(np.abs(ts-event_times[i])) + win_npts[0]
                ind = np.searchsorted(ts_pad, event_times[i])
                et_signal[:, i] = signal[(ind - win_npts[0]): (ind + win_npts[1])]
    return et_signal, et_ts

# eta = event triggered averages: Version: skip events too close to edge
# VERSION: sample window is more likely to be the same in different recordings
def prepare_eta_skip(signal, ts, event_times, win):
    samp_period = np.round((ts[1] - ts[0]), decimals=3)
    win_npts = [np.round(np.abs(win[0])/samp_period).astype(int),
                np.round(np.abs(win[1])/samp_period).astype(int)]
#    win_npts = [ts[ts < ts[0] + np.abs(win[0])].size,
#                ts[ts < ts[0] + np.abs(win[1])].size]
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
    
    
# definition for self_calculated variance
def MADAM(data_pts, descriptor):
    v = np.sum(np.abs(data_pts-descriptor))/data_pts.size
    return v



# %% Load data

dataset_folder = (r'C:\Users\akees\Documents\Ashley\Papers\MIND 1\Cell Reports\Dryad upload\Dataset')

cell_files = os.listdir(dataset_folder)
data = [{} for k in np.arange(len(cell_files))]
for i in np.arange(len(cell_files)):
    full_file = os.path.join(dataset_folder, cell_files[i])
    data[i] = np.load(full_file, allow_pickle=True).item()

states = [{'state':'theta', 'bef':-2.5, 'aft':0.5, 'samp_time':2,
           't_win':[-30, 30], 'd_win':[-4, 12]},
          {'state':'LIA', 'bef':-4, 'aft':-1, 'samp_time':2,
           't_win':[-30, 30], 'd_win':[-4, 12]},
          {'state':'run_theta', 'bef':-2.5, 'aft':0.5, 'samp_time':2,
           't_win':[-30, 30], 'd_win':[-4, 12]},
          {'state':'nonrun_theta', 'bef':-2.5, 'aft':0.5, 'samp_time':2,
           't_win':[-30, 30], 'd_win':[-4, 12]}]




# %% process data

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
        c_p = np.zeros(data[i][states[l]['state']+'_start'].size)
        Ih = np.zeros(data[i][states[l]['state']+'_start'].size)
        Vm0 = np.zeros(data[i][states[l]['state']+'_start'].size)
        dVm = np.zeros(data[i][states[l]['state']+'_start'].size)
        dVm_p = np.zeros(data[i][states[l]['state']+'_start'].size)
        for j in np.arange(data[i][states[l]['state']+'_start'].size):
            # find indices
            bef_ind = int(np.sum(data[i]['Vm_ds_ts'] <
                          (data[i][states[l]['state']+'_start'][j] + states[l]['bef'])))
            aft_ind = int(np.sum(data[i]['Vm_ds_ts'] <
                          (data[i][states[l]['state']+'_start'][j] + states[l]['aft'])))
            # put nan if times are straddling a time when dIh is changed
            dIh_true = np.where((dIh_ind > bef_ind) & (dIh_ind < aft_ind + num_ind))[0]
            if dIh_true.size > 0:
                Ih[j] = np.nan
                Vm0[j] = np.nan
                dVm[j] = np.nan
                dVm_p[j] = np.nan
            else:
                if np.logical_or(l==0, l==1):
                    c_p[j] = data[i][states[l]['state']+'_cell_p']
                else:
                    c_p[j] = data[i]['theta_cell_p']
                Ih_ind = np.searchsorted(data[i]['Vm_Ih_ts'],
                                         data[i][states[l]['state']+'_start'][j])
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
        data[i][states[l]['state']+'_c_p'] = c_p
        data[i][states[l]['state']+'_Ih'] = Ih
        data[i][states[l]['state']+'_Vm0'] = Vm0
        data[i][states[l]['state']+'_dVm'] = dVm
        data[i][states[l]['state']+'_dVm_p'] = dVm_p
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
        dVm_win = [states[l]['bef'], states[l]['aft']+states[l]['samp_time']]
        t_Vm, t_ts = prepare_eta_keep(data[i]['Vm_s_ds'], data[i]['Vm_ds_ts'],
                                 data[i][states[l]['state']+'_start'],
                                 states[l]['t_win'], dVm_win)
        t_sp = prepare_eta_times(data[i]['sp_times'],
                                 data[i][states[l]['state']+'_start'],
                                 states[l]['t_win'])
        data[i][states[l]['state']+'_Vm'] = t_Vm
        data[i][states[l]['state']+'_sp'] = t_sp
    states[l]['t_ts'] = t_ts 

## add windows triggered by offset of some brain states
#for l in np.arange(len(states)):
#    for i in np.arange(len(data)):
#        dVm_win = [states[l]['bef'], states[l]['aft']+states[l]['samp_time']]
#        t_Vm, t_ts = prepare_eta_keep(data[i]['Vm_s_ds'], data[i]['Vm_ds_ts'],
#                                 data[i][states[l]['state']+'_stop'],
#                                 states[l]['t_win'], dVm_win)
#        t_sp = prepare_eta_times(data[i]['sp_times'],
#                                 data[i][states[l]['state']+'_start'],
#                                 states[l]['t_win'])
#        data[i][states[l]['state']+'_Vm'] = t_Vm
#        data[i][states[l]['state']+'_sp'] = t_sp
#    states[l]['t_ts'] = t_ts 


# add triggered windows to event dictionary
# VERSION: only removes events with all nans (not any nans)
for l in np.arange(len(events)):
    raster_sp = []
    psth_sp = np.empty(0)
    Vm = np.empty((states[l]['t_ts'].shape[0], 0))
    duration = np.empty(0)
    cell_id = np.empty(0)
    for i in np.arange(len(data)):
        cell_psth_sp = np.empty(0)
        if data[i][states[l]['state']+'_start'].size > 0:
            Vm = np.append(Vm, data[i][states[l]['state']+'_Vm'], axis=1)
            duration = np.append(duration, (data[i][states[l]['state']+'_stop'] -
                                     data[i][states[l]['state']+'_start']))
            if isinstance(data[i]['cell_id'], str):
                ind = data[i]['cell_id'].find('_')
                cell_int = int(data[i]['cell_id'][:ind])*np.ones(data[i][states[l]['state']+'_start'].size)
                cell_id = np.append(cell_id, cell_int)
            else:
                cell_int = data[i]['cell_id']*np.ones(data[i][states[l]['state']+'_start'].size)
                cell_id = np.append(cell_id, cell_int)
            for j in np.arange(data[i][states[l]['state']+'_start'].size):
                psth_sp = np.append(psth_sp, data[i][states[l]['state']+'_sp'][j])
                cell_psth_sp = np.append(cell_psth_sp, data[i][states[l]['state']+'_sp'][j])
                raster_sp.append(data[i][states[l]['state']+'_sp'][j])
            data[i][states[l]['state']+'_psth_sp'] = cell_psth_sp
    # remove nans
    no_nan = np.logical_and([~np.isnan(Vm).all(axis=0)],
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

# normalize the Vm to start of the event-triggered window
for l in np.arange(len(events)):
    start_ind = np.sum(states[l]['t_ts'] < states[l]['bef'])
    stop_ind = np.sum(states[l]['t_ts'] < (states[l]['bef'] + states[l]['samp_time']))
    comp = np.mean(events[l]['Vm'][start_ind:stop_ind], axis=0)
    events[l]['norm_Vm'] = events[l]['Vm'] - comp




# for each event, label with -1 (hyp), 0 (no), 1 (dep)
for l in np.arange(len(events)):
    dVm_type = np.zeros(events[l]['dVm'].size)
    dVm_type[np.logical_and(events[l]['dVm']<0, events[l]['dVm_p']<0.05)] = -1
    dVm_type[np.logical_and(events[l]['dVm']>0, events[l]['dVm_p']<0.05)] = 1
    events[l]['dVm_type'] = dVm_type

# take the norm_Vm matrices and realign them to the offsets
for l in np.arange(len(events)):
    d_win = -1*np.array(states[l]['d_win'][::-1])
    win_ind = np.searchsorted(states[l]['t_ts'], d_win)
    states[l]['t_ts_off'] = states[l]['t_ts'][win_ind[0]:win_ind[1]]
    ind0 = np.searchsorted(states[l]['t_ts'], 0)
    off_norm_Vm = np.full((int(np.diff(win_ind)), events[l]['duration'].size), np.nan)
    for j in np.arange(events[l]['duration'].size):
        shift_ind = np.searchsorted(states[l]['t_ts'], events[l]['duration'][j]) - ind0
        off_norm_Vm[:, j] = events[l]['norm_Vm'][win_ind[0]+shift_ind:win_ind[1]+shift_ind, j]
    events[l]['off_norm_Vm'] = off_norm_Vm


# %% make average spectrograms

#l = 0
#spec_win = [-3, 3]
#
## triggered z_Sxx for theta onset
#t_Sxx, t_spec_ts = prepare_eta_skip(data[0]['z_Sxx'], data[0]['spec_ts'],
#                               data[0]['theta_start'], spec_win)
#t_num_pts_spec = t_spec_ts.shape[0]
#f_num_pts = data[0]['z_Sxx'].shape[0]
#all_t_Sxx = np.empty([f_num_pts, t_num_pts_spec, 0])
#for i in np.arange(len(data)):
#    t_Sxx, t_spec_ts = prepare_eta_skip(data[i]['z_Sxx'], data[i]['spec_ts'],
#                                   data[i]['theta_start'], spec_win)
#    if t_Sxx.size > 0:
#        all_t_Sxx = np.append(all_t_Sxx, t_Sxx, axis=2)
#mean_on_Sxx = np.mean(all_t_Sxx, axis=2)
## triggered z_Sxx for theta offset
#t_Sxx, t_spec_ts = prepare_eta_skip(data[0]['z_Sxx'], data[0]['spec_ts'],
#                               data[0]['theta_stop'], spec_win)
#t_num_pts_spec = t_spec_ts.shape[0]
#f_num_pts = data[0]['z_Sxx'].shape[0]
#all_t_Sxx = np.empty([f_num_pts, t_num_pts_spec, 0])
#for i in np.arange(len(data)):
#    t_Sxx, t_spec_ts = prepare_eta_skip(data[i]['z_Sxx'], data[i]['spec_ts'],
#                                   data[i]['theta_stop'], spec_win)
#    if t_Sxx.size > 0:
#        all_t_Sxx = np.append(all_t_Sxx, t_Sxx, axis=2)
#mean_off_Sxx = np.mean(all_t_Sxx, axis=2)


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
c_state = [c_run_theta, c_LIA, c_mgry]

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
fig_folder = r'C:\Users\akees\Documents\Ashley\Figures\2020-05_Paper_MIND1\Fig3'


# %% Make figures


dVm = ['hyp', 'no', 'dep']

# plot the dVm color plot, events organized by event duration
# version: blue/red for hyp/dep; hyp/dep/no events plotted on separated figures
l = 0
m = 0
fig, ax = plt.subplots(1, 1, figsize=[1.8, 3.45])
# transpose the norm Vm
norm_Vm = np.transpose(events[l]['norm_Vm'][:, events[l]['dVm_type'] == m-1])
duration = events[l]['duration'][events[l]['dVm_type'] == m-1]
# set order
order = np.flip(np.argsort(duration), axis=0)
p = ax.pcolormesh(states[l]['t_ts'], np.arange(order.size),
                  norm_Vm[order], cmap='RdBu_r', vmin=-5, vmax=5)
ax.scatter(duration[order], np.arange(order.size)+0.5,
        color=c_blk, s=1)
ax.scatter(np.zeros(order.size), np.arange(order.size)+0.5,
        color=c_blk, s=1) 
ax.axis('tight')
ax.set_xlim(states[l]['d_win'])
ax.set_xticks([-4, 0, 4, 8, 12])
ax.set_yticks([order.size-1])
ax.set_yticklabels([order.size])
ax.set_ylim([0, order.size-1])
ax.set_ylabel('events', verticalalignment='center')
ax.yaxis.set_label_coords(-0.1, 0.5, transform=None)
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.set_xlabel('time relative to theta\nonset (s)')
fig.tight_layout()
plt.savefig(os.path.join(fig_folder, 'Vm_color_' + dVm[m] + '.png'),
            transparent=True)


m = 2
fig, ax = plt.subplots(1, 1, figsize=[1.8, 2.1])
# transpose the norm Vm
norm_Vm = np.transpose(events[l]['norm_Vm'][:, events[l]['dVm_type'] == m-1])
duration = events[l]['duration'][events[l]['dVm_type'] == m-1]
# set order
order = np.flip(np.argsort(duration), axis=0)
p = ax.pcolormesh(states[l]['t_ts'], np.arange(order.size),
                  norm_Vm[order], cmap='RdBu_r', vmin=-5, vmax=5)
ax.scatter(duration[order], np.arange(order.size)+0.5,
        color=c_blk, s=1)
ax.scatter(np.zeros(order.size), np.arange(order.size)+0.5,
        color=c_blk, s=1) 
ax.axis('tight')
ax.set_xlim(states[l]['d_win'])
ax.set_xticks([-4, 0, 4, 8, 12])
ax.set_yticks([order.size-1])
ax.set_yticklabels([order.size])
ax.set_ylim([0, order.size-1])
ax.set_ylabel('events', verticalalignment='center')
ax.yaxis.set_label_coords(-0.1, 0.5, transform=None)
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.set_xlabel('time relative to theta\nonset (s)')
# add a scale bar for the colors
divider = make_axes_locatable(ax)
cax = divider.append_axes("top", size="10%", pad=0.1)
cb = plt.colorbar(p, cax=cax, orientation="horizontal", ticks=[-5, 5])
cb.set_label(r'$\Delta$'+' Vm (mV)', labelpad=-22)
axcb = cb.ax
axcb.tick_params(bottom=False, top=True, labelbottom=False, labeltop=True)
#axcb.text(0, 15, r'$\Delta$'+' Vm (mV)', rotation=0, horizontalalignment='center')
fig.tight_layout()
plt.savefig(os.path.join(fig_folder, 'Vm_color_' + dVm[m] + '.png'),
            transparent=True)


#%% make figures for on/offset kinetics


## avg spectrograms and on/offset kinetics
# VERSION: hyp and dep traces on same axis
# onset kinetics
fig, ax = plt.subplots(2, 2, figsize=[3.4, 3.5], sharex='col', sharey='row',
                       gridspec_kw = {'height_ratios':[1, 2]})
## average spectrogram - onset
#im = ax[0, 0].pcolormesh(t_spec_ts, data[0]['f'][data[0]['f']<16],
#                    mean_on_Sxx[data[0]['f']<16],
#                    shading='flat', cmap='viridis', vmin=-0.5, vmax=0.5)
#ax[0, 0].axvline(0, linestyle='--', color=c_blk)
#ax[0, 0].axis('tight')
#ax[0, 0].set_yticks([8, 16])
#ax[0, 0].spines['top'].set_visible(True)
#ax[0, 0].spines['right'].set_visible(True) 
#divider = make_axes_locatable(ax[0, 0])
#cax = divider.append_axes("top", size="10%", pad=0.1)
#cb = plt.colorbar(im, cax=cax, orientation="horizontal", ticks=[-0.5, 0.5])
#cb.set_label('power (z)', labelpad=-21)
#axcb = cb.ax
#axcb.tick_params(bottom=False, top=True, labelbottom=False, labeltop=True)
## average spectrogram - offset
#im = ax[0, 1].pcolormesh(t_spec_ts, data[0]['f'][data[0]['f']<16],
#                    mean_off_Sxx[data[0]['f']<16],
#                    shading='flat', cmap='viridis', vmin=-0.5, vmax=0.5)
#ax[0, 1].axvline(0, linestyle='--', color=c_blk)
#ax[0, 1].axis('tight')
#ax[0, 1].set_yticks([8, 16])
#ax[0, 1].spines['top'].set_visible(True)
#ax[0, 1].spines['right'].set_visible(True) 
#divider = make_axes_locatable(ax[0, 1])
#cax = divider.append_axes("top", size="10%", pad=0.1)
#cb = plt.colorbar(im, cax=cax, orientation="horizontal", ticks=[-0.5, 0.5])
#cb.set_label('power (z)', labelpad=-21)
#axcb = cb.ax
#axcb.tick_params(bottom=False, top=True, labelbottom=False, labeltop=True)
# average hyp - onset
m = 0
mean_Vm = np.nanmean(events[l]['norm_Vm'][:, events[l]['dVm_type'] == m-1], axis=1)
sem_Vm = stats.sem(events[l]['norm_Vm'][:, events[l]['dVm_type'] == m-1], axis=1, nan_policy='omit')
ax[1, 0].fill_between(states[l]['t_ts'], (mean_Vm + sem_Vm), (mean_Vm - sem_Vm),
                    facecolor=c_hyp, linewidth=0, alpha=0.5, zorder=1)
ax[1, 0].plot(states[0]['t_ts'], mean_Vm, color=c_hyp, zorder=4)
ax[1, 0].axhline(0, linestyle='--', color=c_blk)
ax[1, 0].axvline(0, linestyle='--', color=c_blk)
# average hyp - offset
m = 0
mean_Vm = np.nanmean(events[l]['off_norm_Vm'][:, events[l]['dVm_type'] == m-1], axis=1)
sem_Vm = stats.sem(events[l]['off_norm_Vm'][:, events[l]['dVm_type'] == m-1], axis=1, nan_policy='omit')
ax[1, 1].fill_between(states[0]['t_ts_off'], (mean_Vm + sem_Vm), (mean_Vm - sem_Vm),
                    facecolor=c_hyp, linewidth=0, alpha=0.5, zorder=1)
ax[1, 1].plot(states[0]['t_ts_off'], mean_Vm, color=c_hyp, zorder=4)
ax[1, 1].axhline(0, linestyle='--', color=c_blk)
ax[1, 1].axvline(0, linestyle='--', color=c_blk)
# average dep - onset
m = 2
mean_Vm = np.nanmean(events[l]['norm_Vm'][:, events[l]['dVm_type'] == m-1], axis=1)
sem_Vm = stats.sem(events[l]['norm_Vm'][:, events[l]['dVm_type'] == m-1], axis=1, nan_policy='omit')
ax[1, 0].fill_between(states[l]['t_ts'], (mean_Vm + sem_Vm), (mean_Vm - sem_Vm),
                    facecolor=c_dep, linewidth=0, alpha=0.5, zorder=1)
ax[1, 0].plot(states[0]['t_ts'], mean_Vm, color=c_dep, zorder=4)
ax[1, 0].axhline(0, linestyle='--', color=c_blk)
ax[1, 0].axvline(0, linestyle='--', color=c_blk)
# average dep - offset
m = 2
mean_Vm = np.nanmean(events[l]['off_norm_Vm'][:, events[l]['dVm_type'] == m-1], axis=1)
sem_Vm = stats.sem(events[l]['off_norm_Vm'][:, events[l]['dVm_type'] == m-1], axis=1, nan_policy='omit')
ax[1, 1].fill_between(states[l]['t_ts_off'], (mean_Vm + sem_Vm), (mean_Vm - sem_Vm),
                    facecolor=c_dep, linewidth=0, alpha=0.5, zorder=1)
ax[1, 1].plot(states[l]['t_ts_off'], mean_Vm, color=c_dep, zorder=4)
ax[1, 1].axhline(0, linestyle='--', color=c_blk)
ax[1, 1].axvline(0, linestyle='--', color=c_blk)
# format
ax[1, 0].set_xlim([-2, 1.5])
ax[1, 1].set_xlim([-1.5, 2])
ax[1, 0].set_xticks([-2, -1, 0, 1])
ax[1, 1].set_xticks([-1, 0, 1, 2])
ax[1, 0].set_ylim([-3.6, 3.1])
ax[0, 1].tick_params(left=False, right=True)
ax[1, 1].spines['left'].set_visible(False)
ax[1, 1].spines['right'].set_visible(True)
ax[1, 1].tick_params(left=False, right=True)
ax[0, 0].set_ylabel('Hz', rotation=0, verticalalignment='center')
ax[1, 0].set_ylabel(r'$\Delta$'+' Vm (mV)')
ax[1, 0].set_xlabel('time relative to theta\nonset (s)')
ax[1, 1].set_xlabel('time relative to theta\noffset (s)')
ax[0, 0].set_yticks([8, 16])
fig.tight_layout()
plt.savefig(os.path.join(fig_folder, 'avg_on_off_v2.png'),
            transparent=True)

