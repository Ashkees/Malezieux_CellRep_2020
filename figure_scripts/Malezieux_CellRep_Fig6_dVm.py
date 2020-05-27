# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 14:56:30 2020

@author: Ashley
"""



# Manuscript Malezieux, Kees, Mulle submitted to Cell Reports
# Figure 6
# Description: mechanisms of theta hyperpolarization: dVm vs Vm0



# %% import modules

import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib as mpl
from itertools import compress
from mpl_toolkits.axes_grid1 import Divider, Size
from mpl_toolkits.axes_grid1.mpl_axes import Axes


# %% definitions

## some definitions

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


# %% load data

dataset_folder = (r'C:\Users\akees\Documents\Ashley\Papers\MIND 1\Cell Reports\Dryad upload\Dataset')

cell_files = os.listdir(dataset_folder)
data = [{} for k in np.arange(len(cell_files))]
for i in np.arange(len(cell_files)):
    full_file = os.path.join(dataset_folder, cell_files[i])
    data[i] = np.load(full_file, allow_pickle=True).item()

states = [{'id':'theta', 'bef':-2.5, 'aft':0.5, 'samp_time':2, 't_win':[-3, 3]},
          {'id':'LIA', 'bef':-4, 'aft':-1, 'samp_time':2, 't_win':[-4, 2]}]

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
        dVm_win = [states[l]['bef'], states[l]['aft']+states[l]['samp_time']]
        t_Vm, t_ts = prepare_eta_keep(data[i]['Vm_s_ds'], data[i]['Vm_ds_ts'],
                                 data[i][states[l]['id']+'_start'],
                                 states[l]['t_win'], dVm_win)
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
    no_nan = np.logical_and([~np.isnan(Vm).all(axis=0)],
                            [~np.isnan(events[l]['Vm0'])]).flatten()
#    no_nan = np.logical_and([~np.isnan(Vm).any(axis=0)],
#                            [~np.isnan(events[l]['Vm0'])]).flatten()
    events[l]['Vm'] = Vm[:, no_nan]
    events[l]['cell_id'] = cell_id[no_nan]
    events[l]['duration'] = duration[no_nan]
    events[l]['raster_sp'] = list(compress(raster_sp, no_nan))
    events[l]['c_p'] = events[l]['c_p'][no_nan]
    events[l]['Ih'] = events[l]['Ih'][no_nan]
    events[l]['Vm0'] = events[l]['Vm0'][no_nan]
    events[l]['dVm'] = events[l]['dVm'][no_nan]
    events[l]['dVm_p'] = events[l]['dVm_p'][no_nan]



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
c_state = [c_run_theta, c_lbwn, c_mgry]

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
fig_folder = r'C:\Users\akees\Documents\Ashley\Figures\2020-05_Paper_MIND1\Fig6'


# %% make figures

l = 0

# scatter dVm vs Vm0 for theta and LIA
# prep the bools to separate events with and without holding current
Ih0 = (np.abs(events[l]['Ih'])<10)
# make the scatter
#fig, ax = plt.subplots(1, 1, figsize=[2.2, 1.8])
# create a figure with axes of defined size
fig = plt.figure(figsize=[2, 2])
# The first items are for padding and the second items are for the axes.
# sizes are in inch.
h = [Size.Fixed(0.5), Size.Fixed(1.3)]
v = [Size.Fixed(0.5), Size.Fixed(1.1)]
divider = Divider(fig, (0.0, 0.0, 1., 1.), h, v, aspect=False)
ax = Axes(fig, divider.get_position())
ax.set_axes_locator(divider.new_locator(nx=1, ny=1))
fig.add_axes(ax)
ax.scatter(events[l]['Vm0'][Ih0],
           events[l]['dVm'][Ih0],
           s=10, facecolors=c_state[l], edgecolors=c_state[l])
ax.set_ylim([-15, 15])
ax.set_xlim([-70, -20])
ax.set_yticks([-15, 0, 15])
ax.set_xticks([-60, -40, -20])
ax.tick_params(top=False, right=False, length=6)
ax.spines['left'].set_bounds(-15, 15)
ax.set_xlabel('initial Vm (mV)')
ax.set_ylabel(r'$\Delta$ Vm (mV)', labelpad=0)
#fig.tight_layout()
plt.savefig(os.path.join(fig_folder, 'dVm_Vm0_'+ states[l]['id']+'.png'),
                         transparent=True)



# make the example traces
#fig, ax = plt.subplots(1, 1, figsize=[1.5, 1.8], sharey=True)
# create a figure with axes of defined size
fig = plt.figure(figsize=[2, 2])
# The first items are for padding and the second items are for the axes.
# sizes are in inch.
h = [Size.Fixed(0.5), Size.Fixed(0.8)]
v = [Size.Fixed(0.5), Size.Fixed(1.1)]
divider = Divider(fig, (0.0, 0.0, 1., 1.), h, v, aspect=False)
ax = Axes(fig, divider.get_position())
ax.set_axes_locator(divider.new_locator(nx=1, ny=1))
fig.add_axes(ax)
if l == 0:
    i = 14  # cell 56
#if l == 1:
#    i = 7  # cell 87
ax.plot(states[l]['t_ts'], data[i][states[l]['id']+'_Vm'], linewidth=1,
           color=c_blk)
ax.axvspan(0, 5, ymin=0, ymax=1, color=c_state[l], alpha=0.5)
if l == 0:
    ax.set_xlim([-2, 2])
#if l == 1:
#    ax[l].set_xlim([-3, 1])
ax.set_xlabel('time relative to\ntheta onset (s)')
ax.set_ylabel('Vm (mV)')
ax.set_title('Cell 10', loc='left', fontsize=8)
#fig.tight_layout()
plt.savefig(os.path.join(fig_folder, 'dVm_'+ states[l]['id']+'_ex.png'),
                         transparent=True)





