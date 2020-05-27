# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 14:46:53 2019

@author: Ashley
"""


# Manuscript Malezieux, Kees, Mulle submitted to Cell Reports
# Figure S7
# Description: time lag between change in Vm and change in variance



# %% import modules

import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib as mpl


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


# eta = event triggered averages
# VERSION: store only spike indices
def prepare_eta_waves(wave_times, event_times, win):
    et_signal = []
    et_waves_ind = []
    if (wave_times.size > 0) & (event_times.size > 0):
        # find wave_times that occur within window of each event_time
        for i in np.arange(event_times.size):
            waves_ind = ((wave_times > event_times[i] + win[0]) &
                         (wave_times < event_times[i] + win[1]))
            ts_section = wave_times[waves_ind]
            ts_section = ts_section - event_times[i]
            et_signal.append(ts_section)
            et_waves_ind.append(np.where(waves_ind)[0]) 
    else:
        et_signal = [np.empty(0) for k in np.arange(event_times.size)]
        et_waves_ind = [np.empty(0) for k in np.arange(event_times.size)]
    return et_signal, et_waves_ind


# definition for finding run times
def find_dVm_times(Vm_ts, dVm, win):
    nonzero_dVm = np.array(dVm > 0, float)
    dVm_start = Vm_ts[np.ediff1d(nonzero_dVm, to_begin=0) == 1]
    dVm_stop = Vm_ts[np.ediff1d(nonzero_dVm, to_begin=0) == -1]
    if np.logical_or(dVm_start.size==0, dVm_stop.size==0):
        dVm_start = np.empty(0)
        dVm_stop = np.empty(0)
    else:
        # remove runs that occur either at very start or very end of recording
        if dVm_start[0]-dVm_stop[0] > 0:
            dVm_stop = dVm_stop[1:]
        if dVm_start.shape != dVm_stop.shape:
            dVm_start = dVm_start[0:-1]
        while (dVm_start[0]+win[0])<Vm_ts[0]:
            dVm_start = dVm_start[1:]
            dVm_stop = dVm_stop[1:]
        while (dVm_stop[-1]+win[1])>Vm_ts[-1]:
            dVm_start = dVm_start[:-1]
            dVm_stop = dVm_stop[:-1]
    return dVm_start, dVm_stop 


# %% load data

dataset_folder = (r'C:\Users\akees\Documents\Ashley\Papers\MIND 1\Cell Reports\Dryad upload\Dataset')

cell_files = os.listdir(dataset_folder)
data = [{} for k in np.arange(len(cell_files))]
for i in np.arange(len(cell_files)):
    full_file = os.path.join(dataset_folder, cell_files[i])
    data[i] = np.load(full_file, allow_pickle=True).item()

# %% process data

# for each cell, detect hyp and dep in Vm in the same manner as fig 3 and 4
# step through every 500ms (10 datapoints) and determine whether there is a
# significant change in Vm
num_ind = 40
t_win = [-20, 100]
for i in np.arange(len(data)):
    samp_freq = 1/(data[i]['Vm_ds_ts'][1] - data[i]['Vm_ds_ts'][0])
    test_ind = np.arange(50, data[i]['Vm_ds'].size-50, 10)
    dep_times = np.zeros(test_ind.size)
    dep_dVm = np.zeros(test_ind.size)
    hyp_times = np.zeros(test_ind.size)
    hyp_dVm = np.zeros(test_ind.size)
    # find index of dIh_times
    dIh_ind = data[i]['dIh_times']*samp_freq
    dIh_ind = dIh_ind.astype(int)
    for j in np.arange(test_ind.size):
        # find indices
        bef_ind = test_ind[j]-50
        aft_ind = test_ind[j]+10
        # ignore times where Ih is changed
        dIh_true = np.where((dIh_ind > bef_ind-num_ind) & (dIh_ind < aft_ind + 2*num_ind))[0]
        if dIh_true.size == 0:
            # test whether Vm values are significantly different
            # Welch's t-test: normal, unequal variances, independent samp
            t, p = stats.ttest_ind(data[i]['Vm_ds'][bef_ind:bef_ind+num_ind],
                                   data[i]['Vm_ds'][aft_ind:aft_ind+num_ind],
                                   equal_var=False, nan_policy='omit')
            if p<0.05:
                if (np.nanmean(data[i]['Vm_ds'][aft_ind:aft_ind+num_ind]) - 
                    np.nanmean(data[i]['Vm_ds'][bef_ind:bef_ind+num_ind])) > 0:
                    dep_times[j] = data[i]['Vm_ds_ts'][test_ind[j]]
                    dep_dVm[j] = (np.nanmax(data[i]['Vm_s_ds'][aft_ind:aft_ind+num_ind]) - 
                                  np.nanmin(data[i]['Vm_s_ds'][bef_ind:bef_ind+num_ind]))
                else:
                    hyp_times[j] = data[i]['Vm_ds_ts'][test_ind[j]]
                    hyp_dVm[j] = (np.nanmin(data[i]['Vm_s_ds'][aft_ind:aft_ind+num_ind]) - 
                                  np.nanmax(data[i]['Vm_s_ds'][bef_ind:bef_ind+num_ind]))
    dep_start, dep_stop = find_dVm_times(test_ind, dep_dVm, t_win)
    hyp_start, hyp_stop = find_dVm_times(test_ind, -1*hyp_dVm, t_win)
    data[i]['dep_start'] = data[i]['Vm_ds_ts'][dep_start]
    data[i]['dep_stop'] = data[i]['Vm_ds_ts'][dep_stop]
    data[i]['hyp_start'] = data[i]['Vm_ds_ts'][hyp_start]
    data[i]['hyp_stop'] = data[i]['Vm_ds_ts'][hyp_stop]

# add windows triggered by start of hyperpolarizations and depolarizations
dVms = ['dep', 'hyp']
t_win = [-1, 5]
for l in np.arange(len(dVms)):
    for i in np.arange(len(data)):
        t_Vm, t_ts = prepare_eta(data[i]['Vm_s_ds'], data[i]['Vm_ds_ts'],
                                 data[i][dVms[l]+'_start'], t_win)
        t_var, t_ts = prepare_eta(data[i]['Vm_var'], data[i]['Vm_ds_ts'],
                                 data[i][dVms[l]+'_start'], t_win)
        data[i][dVms[l]+'_Vm'] = t_Vm
        data[i][dVms[l]+'_var'] = t_var

# find which dep and hyp coincide with theta and LIA starts
dVms = ['hyp', 'dep']
states = ['theta', 'LIA']
for i in np.arange(len(data)):
    for m in np.arange(len(dVms)):
        for l in np.arange(len(states)):
            if l == 0:
                sh = 0
            if l == 1:
                sh = -1.5
            temp_bool = np.zeros(data[i][dVms[m]+'_start'].size)
            temp_times = np.empty(0)
            for j in np.arange(data[i][dVms[m]+'_start'].size):
                state_true = data[i][states[l]+'_start'][(data[i][states[l]+'_start']+sh+0.20>=data[i][dVms[m]+'_start'][j]) &
                              (data[i][states[l]+'_start']+sh-0.20<=data[i][dVms[m]+'_stop'][j])]
                if state_true.size>0:
                    temp_bool[j] = 1
                    temp_times = np.append(temp_times, state_true) 
            data[i][states[l]+'_'+dVms[m]] = temp_bool
            data[i][states[l]+'_'+dVms[m]+'_times'] = temp_times

# put all the events together
events = [{'event': 'hyp'}, {'event': 'dep'}]

for m in np.arange(len(events)):
    Vm = np.empty((t_ts.shape[0], 0))
    var = np.empty((t_ts.shape[0], 0))
    duration = np.empty(0)
    theta_bool = np.empty(0)
    LIA_bool = np.empty(0)
    cell_id = np.empty(0)
    theta_cell_p = np.empty(0)
    LIA_cell_p = np.empty(0)
    for i in np.arange(len(data)):
        Vm = np.append(Vm, data[i][events[m]['event']+'_Vm'], axis=1)
        var = np.append(var, data[i][events[m]['event']+'_var'], axis=1)
        duration = np.append(duration, (data[i][events[m]['event']+'_stop'] -
                             data[i][events[m]['event']+'_start']))
        theta_bool = np.append(theta_bool, data[i]['theta_'+events[m]['event']])
        LIA_bool = np.append(LIA_bool, data[i]['LIA_'+events[m]['event']])
        if isinstance(data[i]['cell_id'], str):
            ind = data[i]['cell_id'].find('_')
            cell_int = int(data[i]['cell_id'][:ind])*np.ones(data[i][events[m]['event']+'_start'].size)
            cell_id = np.append(cell_id, cell_int)
        else:
            cell_int = data[i]['cell_id']*np.ones(data[i][events[m]['event']+'_start'].size)
            cell_id = np.append(cell_id, cell_int)
        theta_cell_p = np.append(theta_cell_p, data[i]['theta_cell_p']*np.ones(data[i][events[m]['event']+'_start'].size))
        LIA_cell_p = np.append(LIA_cell_p, data[i]['LIA_cell_p']*np.ones(data[i][events[m]['event']+'_start'].size))
    events[m]['Vm'] = Vm
    events[m]['var'] = var
    events[m]['duration'] = duration
    events[m]['cell_id'] = cell_id
    events[m]['theta_cell_p'] = theta_cell_p
    events[m]['LIA_cell_p'] = LIA_cell_p
    events[m]['theta_bool'] = np.array(theta_bool, dtype=bool)
    events[m]['LIA_bool'] = np.array(LIA_bool, dtype=bool)
    events[m]['unlabeled_bool'] = np.squeeze(np.array([(theta_bool+LIA_bool) ==
                                                       0], dtype=bool))

# time lag between peak (hyp) or trough (dep) in individual Vm and var
# add the individual time deltas for each event
samp_period = t_ts[1] - t_ts[0]
events[0]['lag'] = (np.argmax(events[0]['var'][0:60, :], axis=0) -
                    np.argmax(events[0]['Vm'][0:60, :], axis=0))*samp_period
events[1]['lag'] = (np.argmin(events[1]['var'][0:60, :], axis=0) -
                    np.argmin(events[1]['Vm'][0:60, :], axis=0))*samp_period

# find the true dVm and dvar for each event
# also find initial (0) and final (1) absolute Vm and var
# in each cell, take Vm and var at hyp/dep start/stop
for m in np.arange(len(events)):
    for i in np.arange(len(data)):
        dVm = np.zeros(data[i][events[m]['event']+'_start'].size)
        Vm0 = np.zeros(data[i][events[m]['event']+'_start'].size)
        Vm1 = np.zeros(data[i][events[m]['event']+'_start'].size)
        dvar = np.zeros(data[i][events[m]['event']+'_start'].size)
        var0 = np.zeros(data[i][events[m]['event']+'_start'].size)
        var1 = np.zeros(data[i][events[m]['event']+'_start'].size)
        for j in np.arange(data[i][events[m]['event']+'_start'].size):
            bef_ind = np.searchsorted(data[i]['Vm_ds_ts'],
                                      data[i][events[m]['event']+'_start'][j])
            aft_ind = np.searchsorted(data[i]['Vm_ds_ts'],
                                      data[i][events[m]['event']+'_stop'][j])
            dVm[j] = data[i]['Vm_s_ds'][aft_ind] - data[i]['Vm_s_ds'][bef_ind]
            Vm0[j] = data[i]['Vm_s_ds'][bef_ind]
            Vm1[j] = data[i]['Vm_s_ds'][aft_ind]
            dvar[j] = data[i]['Vm_var'][aft_ind] - data[i]['Vm_var'][bef_ind]
            var0[j] = data[i]['Vm_var'][bef_ind]
            var1[j] = data[i]['Vm_var'][aft_ind]
        data[i][events[m]['event']+'_dVm'] = dVm
        data[i][events[m]['event']+'_Vm0'] = Vm0
        data[i][events[m]['event']+'_Vm1'] = Vm1
        data[i][events[m]['event']+'_dvar'] = dvar
        data[i][events[m]['event']+'_var0'] = var0
        data[i][events[m]['event']+'_var1'] = var1
# add cell dVm and dvar to events dictionary
for m in np.arange(len(events)):
    dVm = np.empty(0)
    Vm0 = np.empty(0)
    Vm1 = np.empty(0)
    dvar = np.empty(0)
    var0 = np.empty(0)
    var1 = np.empty(0)
    for i in np.arange(len(data)):
        dVm = np.append(dVm, data[i][events[m]['event']+'_dVm'])
        Vm0 = np.append(Vm0, data[i][events[m]['event']+'_Vm0'])
        Vm1 = np.append(Vm1, data[i][events[m]['event']+'_Vm1'])
        dvar = np.append(dvar, data[i][events[m]['event']+'_dvar'])
        var0 = np.append(var0, data[i][events[m]['event']+'_var0'])
        var1 = np.append(var1, data[i][events[m]['event']+'_var1'])
    events[m]['dVm'] = dVm
    events[m]['Vm0'] = Vm0
    events[m]['Vm1'] = Vm1
    events[m]['dvar'] = dvar
    events[m]['var0'] = var0
    events[m]['var1'] = var1

# make the ratio between dVm and dvar
for m in np.arange(len(events)):
    events[m]['dVm_dvar'] = events[m]['dvar']/events[m]['dVm']
    events[m]['Vm1_var1'] = events[m]['var1']/events[m]['Vm1']
    
    

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
#c_state = [c_mgry, c_run_theta, c_lbwn]
c_state = [c_mgry, c_grn, c_bwn]
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
fig_folder = r'C:\Users\akees\Documents\Ashley\Figures\2020-05_Paper_MIND1\FigS7'
 
    
 # %% panels for the actual figure   

states = ['unlabeled', 'theta', 'LIA'] 
    
 # make histograms of lag times for unlabeled/theta/LIA for hyp/dep
m = 0
bins = np.arange(-2, 4, 0.05)
fig, ax = plt.subplots(1, 1, sharex=True, figsize=[2, 2])
plt.hist([events[m]['lag'][events[m][states[1]+'_bool']],
          events[m]['lag'][events[m][states[2]+'_bool']],
          events[m]['lag'][events[m][states[0]+'_bool']]], bins=bins,
          color=[c_grn, c_bwn, c_mgry], density=True, 
          label=['theta', 'LIA', 'unlabeled'], histtype='step',
          cumulative=True)
ax.set_ylim([0, 1])
ax.set_xlim([-2, 3])
ax.set_yticks([0, 0.5, 1])
ax.set_ylabel('cumulative proportion')
ax.set_xticks([-2, -1, 0, 1, 2, 3])
ax.set_xlabel('delay between change in Vm\nand change in variance (s)')
ax.set_title('hyperpolarizations', loc='left', fontsize=8)
fig.tight_layout()
plt.savefig(os.path.join(fig_folder, 'hist_hyp.png'), transparent=True)  


m = 1
bins = np.arange(-2, 4, 0.05)
fig, ax = plt.subplots(1, 1, sharex=True, figsize=[2, 2])
plt.hist([events[m]['lag'][events[m][states[1]+'_bool']],
          events[m]['lag'][events[m][states[2]+'_bool']],
          events[m]['lag'][events[m][states[0]+'_bool']]], bins=bins,
          color=[c_grn, c_bwn, c_mgry], density=True, 
          label=['theta', 'LIA', 'unlabeled'], histtype='step',
          cumulative=True)
ax.set_ylim([0, 1])
ax.set_xlim([-2, 3])
ax.set_yticks([0, 0.5, 1])
ax.set_ylabel('cumulative proportion')
ax.set_xticks([-2, -1, 0, 1, 2, 3])
ax.set_xlabel('delay between change in Vm\nand change in variance (s)')
ax.set_title('depolarizations', loc='left', fontsize=8)
fig.tight_layout()   
plt.savefig(os.path.join(fig_folder, 'hist_dep.png'), transparent=True) 



# bootstrap: test each group to see if lag is different than zero
states = ['unlabeled', 'theta', 'LIA']
real_d = np.zeros([len(events), len(states)])
p = np.zeros([len(events), len(states)])
num_b = 1000
faux_d = np.zeros([len(events), len(states), num_b])
for m in np.arange(len(events)):
    for l in np.arange(len(states)):
        lag = events[m]['lag'][events[m][states[l]+'_bool']]
        #real_d[m, l] = np.nanmean(lag)
        real_d[m, l] = np.nanmedian(lag)
        # change the dataset to have a mean of 0 (demean)
        #lag0 = lag-np.nanmean(lag)
        lag0 = lag-np.nanmedian(lag)
        for b in np.arange(num_b):
            f_select = np.random.randint(0, lag0.size, lag0.size)
            f_lag = lag0[f_select]
            #faux_d[m, l, b] = np.nanmean(f_lag)
            faux_d[m, l, b] = np.nanmedian(f_lag)
        p[m, l] = np.sum(faux_d[m, l, :] > real_d[m, l])/num_b


# bootstrap: test whether groups are different from each other (anova)
states = ['unlabeled', 'theta', 'LIA']
real_d = np.full([len(events), len(states)-1], np.nan)
t_boot_p = np.full([len(events), len(states)-1], np.nan)
real_F = np.full([len(events)], np.nan)
boot_anova_p = np.full([len(events)], np.nan)
p_kw = np.full([len(events)], np.nan)
med = np.full([len(events), len(states)], np.nan)
std = np.full([len(events), len(states)], np.nan)
madam = np.full([len(events), len(states)], np.nan)
n = np.full([len(events), len(states)], np.nan)
num_b = 1000
faux_d = np.zeros([len(events), len(states), num_b])
for m in np.arange(len(events)):
    groups = [None]*len(states)
    for l in np.arange(len(states)):
        groups[l] = events[m]['lag'][events[m][states[l]+'_bool']]
    real_F[m], boot_anova_p[m] = boot_anova(groups, num_b)
    # try the stats test again with a kruskal-wallace (nonparametric 1-way anova)
    H, p_kw[m] = stats.kruskal(groups[0], groups[1], groups[2], nan_policy='omit')
    if boot_anova_p[m] < 0.1:
        for l in np.arange(len(states)-1):
            real_d[m, l], t_boot_p[m, l] = boot_t(groups[0], groups[l+1], num_b)
    # calculate some descriptors of the groups
    for l in np.arange(len(states)):
        med[m, l] = np.nanmedian(groups[l])
        std[m, l] = np.nanstd(groups[l])
        madam[m, l] = MADAM(groups[l], np.nanmedian(groups[l]))
        n[m, l] = groups[l].size

