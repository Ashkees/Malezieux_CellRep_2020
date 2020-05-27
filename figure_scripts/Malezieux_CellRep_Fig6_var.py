# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 14:29:54 2019

@author: Ashley
"""


# Manuscript Malezieux, Kees, Mulle submitted to Cell Reports
# Figure 6
# Description: mechanisms of theta hyperpolarization: variance



# %% import modules


import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import Divider, Size
from mpl_toolkits.axes_grid1.mpl_axes import Axes


# %% definitions


# find timestamps of a state
def find_state_bool(ts, state_start, state_stop):
    state_bool = np.zeros(ts.size, dtype='bool')
    # find timestamps closest to running start and stop times
    for i in np.arange(state_start.size):
        ind_start = np.argmin(np.abs(ts-state_start[i]))
        ind_stop = np.argmin(np.abs(ts-state_stop[i]))
        state_bool[ind_start:ind_stop] = True
    return state_bool


# definition for self_calculated variance (called MADAM??)
def MADAM(data_pts, descriptor):
    v = np.sum(np.abs(data_pts-descriptor))/data_pts.size
    return v


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




def boot_t(t_g0, t_g1, num_b):
    real_d = np.median(t_g1) - np.median(t_g0)
    faux_d = np.zeros(num_b)
    box = np.append(t_g0, t_g1)
    for b in np.arange(num_b):
        f_g0 = box[np.random.randint(0, box.size, size=t_g0.size)]
        f_g1 = box[np.random.randint(0, box.size, size=t_g1.size)]
        faux_d[b] = np.median(f_g1) - np.median(f_g0)
    p = np.sum(np.abs(faux_d) > np.abs(real_d))/num_b
    return real_d, p


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


# eta = event triggered averages
def prepare_eta(signal, ts, event_times, win):
    win_npts = [ts[ts < ts[0] + np.abs(win[0])].size,
                ts[ts < ts[0] + np.abs(win[1])].size]
    et_ts = ts[0:np.sum(win_npts)] - ts[0] + win[0]
    et_signal = np.empty(0)
    # remove any events that are too close to the beginning or end of recording
    if event_times.size > 0:
        event_times = event_times[event_times+win[0] > ts[0]]
        if event_times.size > 0:
            event_times = event_times[event_times+win[1] < ts[-1]]
            if event_times.size > 0:
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
    

# %% Load cell data
    

dataset_folder = (r'C:\Users\akees\Documents\Ashley\Papers\MIND 1\Cell Reports\Dryad upload\Dataset')

cell_files = os.listdir(dataset_folder)
data = [{} for k in np.arange(len(cell_files))]
for i in np.arange(len(cell_files)):
    full_file = os.path.join(dataset_folder, cell_files[i])
    data[i] = np.load(full_file, allow_pickle=True).item()


# %% process data
    


# prepare the var values for each state
for i in np.arange(len(data)):
    theta_bool = find_state_bool(data[i]['Vm_ds_ts'], data[i]['theta_start'],
                               data[i]['theta_stop'])
    LIA_bool = find_state_bool(data[i]['Vm_ds_ts'], data[i]['LIA_start'],
                               data[i]['LIA_stop'])
    data[i]['theta_var'] = data[i]['Vm_var'][theta_bool]
    data[i]['theta_not_var'] = data[i]['Vm_var'][theta_bool == 0]
    data[i]['LIA_var'] = data[i]['Vm_var'][LIA_bool]
    data[i]['LIA_not_var'] = data[i]['Vm_var'][LIA_bool == 0]
    data[i]['theta_Vm'] = data[i]['Vm_ds'][theta_bool]
    data[i]['theta_not_Vm'] = data[i]['Vm_ds'][theta_bool == 0]
    data[i]['LIA_Vm'] = data[i]['Vm_ds'][LIA_bool]
    data[i]['LIA_not_Vm'] = data[i]['Vm_ds'][LIA_bool == 0]



# for each cell, detect hyp and dep in Vm in the same manner as fig 3 and 5
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

#%% set figure parameters

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
fig_folder = r'C:\Users\akees\Documents\Ashley\Figures\2020-05_Paper_MIND1\Fig6'

#%%


# for all cells, make the scatter of var for LIA vs theta - all cells
# color is sig difference from unity line
state = 'theta'
all_dVm = np.array([d[state+'_mean_dVm'] for d in data])[[isinstance(d['cell_id'], int) for d in data]]
all_cell_p = np.array([d[state+'_cell_p'] for d in data])[[isinstance(d['cell_id'], int) for d in data]]
unique_cells = np.array([isinstance(d['cell_id'], int) for d in data])
# prepare the numbers for the scatter
unique_cells = [isinstance(d['cell_id'], int) for d in data]
x = np.array([np.nanmedian(d[state+'_not_var']) for d in data])[unique_cells]
y = np.array([np.nanmedian(d[state+'_var']) for d in data])[unique_cells]
var_p = np.zeros(len(data))
for i in np.arange(len(data)):
    s, p = stats.mannwhitneyu(data[i][state+'_not_var'], data[i][state+'_var'],
                              alternative='two-sided')
    var_p[i] = p
var_p = var_p[unique_cells]
# eliminate any nans
no_nans = np.logical_or(np.isnan(x), np.isnan(y)) == 0
x = x[no_nans]
y = y[no_nans]
var_p = var_p[no_nans]
all_cell_p = all_cell_p[no_nans]
all_dVm = all_dVm[no_nans]
# make the scatter
s_cell = 20
#fig, ax = plt.subplots(1, 1, figsize=[1.75, 1.75])
# create a figure with axes of defined size
fig = plt.figure(figsize=[2.5, 2.5])
# The first items are for padding and the second items are for the axes.
# sizes are in inch.
h = [Size.Fixed(0.75), Size.Fixed(1.4)]
v = [Size.Fixed(0.75), Size.Fixed(1.4)]
divider = Divider(fig, (0.0, 0.0, 1., 1.), h, v, aspect=False)
ax = Axes(fig, divider.get_position())
ax.set_axes_locator(divider.new_locator(nx=1, ny=1))
fig.add_axes(ax)
ax.scatter(x[all_cell_p < 0.05], y[all_cell_p < 0.05], s=10, facecolors='none',
              edgecolors=c_hyp, zorder=3)
ax.scatter(x[(all_cell_p < 0.05) & (var_p <0.05)], y[(all_cell_p < 0.05) & (var_p <0.05)], s=s_cell,
              facecolors=c_hyp, edgecolors=c_hyp, zorder=3)
ax.scatter(x[all_cell_p > 0.95], y[all_cell_p > 0.95], s=s_cell, facecolors='none',
              edgecolors=c_dep, zorder=3)
ax.scatter(x[(all_cell_p > 0.95) & (var_p <0.05)], y[(all_cell_p > 0.95) & (var_p <0.05)], s=s_cell,
              facecolors=c_dep, edgecolors=c_dep, zorder=3)
ax.scatter(x[(all_dVm < 0) & (all_cell_p >= 0.05)], y[(all_dVm < 0) & (all_cell_p >= 0.05)], s=s_cell, facecolors='none',
              edgecolors=c_mgry, zorder=2)
ax.scatter(x[(all_dVm < 0) & (all_cell_p >= 0.05) & (var_p <0.05)], y[(all_dVm < 0) & (all_cell_p >= 0.05) & (var_p <0.05)], s=s_cell,
              facecolors=c_mgry, edgecolors=c_mgry, zorder=2)
ax.scatter(x[(all_dVm > 0) & (all_cell_p <= 0.95)], y[(all_dVm > 0) & (all_cell_p <= 0.95)], s=s_cell, facecolors='none',
              edgecolors=c_mgry, zorder=2)
ax.scatter(x[(all_dVm > 0) & (all_cell_p <= 0.95) & (var_p <0.05)], y[(all_dVm > 0) & (all_cell_p <= 0.95) & (var_p <0.05)], s=s_cell,
              facecolors=c_mgry, edgecolors=c_mgry, zorder=2)
ax.plot([0, 15], [0, 15], color=c_blk, zorder=1)
ax.axis('square')
ax.set_ylim([0, 16])
ax.set_xlim([0, 16])
ax.set_yticks([0, 4, 8, 12, 16])
ax.set_xticks([0, 4, 8, 12, 16])
ax.tick_params(top=False, right=False,length=6)
ax.set_ylabel('Vm variance\nduring theta (mV$^\mathrm{2}$)')
ax.set_xlabel('Vm variance\nduring nontheta (mV$^\mathrm{2}$)')
#fig.tight_layout()
plt.savefig(os.path.join(fig_folder, 'var_vs_Vm_'+state+'.png'), transparent=True)

# stats: paired t-test bootstrap
# each sample is the real differences, but + or - is randomly assigned
diff = y-x
real_diff = np.mean(diff)
# perform the bootstrap
num_b = 1000
f_diff = np.zeros(num_b)
for b in np.arange(num_b):
    sample = np.random.choice([-1, 1], size = diff.size, replace=True)
    f_diff[b] = np.mean(diff*sample)
diff_p = np.sum(f_diff<real_diff)/num_b

np.median(x)
np.std(x)
MADAM(x, np.median(x))
np.median(y)
np.std(y)
MADAM(y, np.median(y))


# %% event-based plots

# average of all events - hyperpolarizations
fig, ax = plt.subplots(2, 1, figsize=[1.4, 1.6])
Vm = events[0]['Vm']
sem_Vm = stats.sem(Vm, axis=1, nan_policy='omit')
ax[0].plot(t_ts, np.nanmean(Vm, axis=1), color=c_blk)
ax[0].fill_between(t_ts, (np.nanmean(Vm, axis=1)+sem_Vm),
                   (np.nanmean(Vm, axis=1)-sem_Vm),
                   facecolor=c_blk, linewidth=0, alpha=0.25)
ax[0].axvspan(0, 2.5, ymin=0, ymax=1, color=c_hyp, alpha=0.25)
ax[0].set_ylim([-53, -48])
ax[0].spines['left'].set_bounds(-53, -48)
ax[0].spines['bottom'].set_visible(False)
ax[0].set_xticks([])
ax[0].set_yticks([-53, -48])
var = events[0]['var']
sem_var = stats.sem(var, axis=1, nan_policy='omit')
ax[1].plot(t_ts, np.nanmean(var, axis=1), color=c_sp)
ax[1].fill_between(t_ts, (np.nanmean(var, axis=1)+sem_var),
                   (np.nanmean(var, axis=1)-sem_var),
                   facecolor=c_sp, linewidth=0, alpha=0.5)
ax[1].axvspan(0, 2.5, ymin=0, ymax=1, color=c_hyp, alpha=0.25)
ax[1].set_ylim([5, 9])
ax[1].spines['left'].set_bounds(5, 9)
ax[1].spines['bottom'].set_bounds(3, 5)
ax[1].set_xticks([])
ax[1].set_yticks([5, 9])
ax[0].set_ylabel('mV', rotation=0, verticalalignment='center')
ax[1].set_ylabel('mV$^\mathrm{2}$', rotation=0, verticalalignment='center', labelpad=10)
ax[1].set_xlabel('2 s', horizontalalignment='center')
ax[1].xaxis.set_label_coords(0.8, -0.15, transform=None)
ax[0].set_title('hyperpolarizations', fontsize=8, horizontalalignment='center')
fig.tight_layout()
plt.savefig(os.path.join(fig_folder, 'avg_hyp.png'), transparent=True)

# average of all events - depolarizations
fig, ax = plt.subplots(2, 1, figsize=[1.4, 1.6])
Vm = events[1]['Vm']
sem_Vm = stats.sem(Vm, axis=1, nan_policy='omit')
ax[0].plot(t_ts, np.nanmean(Vm, axis=1), color=c_blk)
ax[0].fill_between(t_ts, (np.nanmean(Vm, axis=1)+sem_Vm),
                   (np.nanmean(Vm, axis=1)-sem_Vm),
                   facecolor=c_blk, linewidth=0, alpha=0.25)
ax[0].axvspan(0, 2.5, ymin=0, ymax=1, color=c_dep, alpha=0.25)
ax[0].set_ylim([-53, -48])
ax[0].spines['left'].set_bounds(-53, -48)
ax[0].spines['bottom'].set_visible(False)
ax[0].set_xticks([])
ax[0].set_yticks([-53, -48])
var = events[1]['var']
sem_var = stats.sem(var, axis=1, nan_policy='omit')
ax[1].plot(t_ts, np.nanmean(var, axis=1), color=c_sp)
ax[1].fill_between(t_ts, (np.nanmean(var, axis=1)+sem_var),
                   (np.nanmean(var, axis=1)-sem_var),
                   facecolor=c_sp, linewidth=0, alpha=0.5)
ax[1].axvspan(0, 2.5, ymin=0, ymax=1, color=c_dep, alpha=0.25)
ax[1].set_ylim([5, 9])
ax[1].spines['left'].set_bounds(5, 9)
ax[1].spines['bottom'].set_bounds(3, 5)
ax[1].set_xticks([])
ax[1].set_yticks([5, 9])
ax[0].set_ylabel('mV', rotation=0, verticalalignment='center')
ax[1].set_ylabel('mV$^\mathrm{2}$', rotation=0, verticalalignment='center', labelpad=10)
ax[1].set_xlabel('2 s', horizontalalignment='center')
ax[1].xaxis.set_label_coords(0.8, -0.15, transform=None)
ax[0].set_title('depolarizations', fontsize=8, horizontalalignment='center')
fig.tight_layout()
plt.savefig(os.path.join(fig_folder, 'avg_dep.png'), transparent=True)


# %%

# make histograms of different state measurements
measure = 'var1'

m = 0
states = ['unlabeled', 'theta']
if measure == 'var1':
    bins = np.arange(0, 20, 0.01)
if measure == 'Vm1':
    bins = np.arange(-80, -20, 0.01)
fig, ax = plt.subplots(1, 1, sharex=True, figsize=[2, 1.8])
plt.hist([events[m][measure][events[m][states[1]+'_bool']],
          events[m][measure][events[m][states[0]+'_bool']]], bins=bins,
          color=[c_grn, c_mgry], density=True, 
          label=['LIA', 'unlabeled'], histtype='step',
          cumulative=True)
if measure == 'Vm1':
    ax.set_xlim([-80, -30])
    ax.set_xticks([-80, -60, -40])
if measure == 'var1':
    ax.set_xlim([0, 16])
    ax.set_xticks([0, 8, 16])
    ax.set_xlabel('Vm variance (mV$^\mathrm{2}$)')
#if m == 0:
#    ax.legend(frameon=False, loc=2, fontsize=20)
ax.set_ylim([0, 1.05])
ax.spines['left'].set_bounds(0, 1)
ax.set_yticks([0, 0.5, 1])
ax.set_yticklabels([0, '', 1])
ax.spines['left'].set_bounds(0, 1)
ax.set_ylabel('cumulative proportion')
fig.tight_layout()
plt.savefig(os.path.join(fig_folder, 'chist_'+events[m]['event']+'_'+measure+'.png'), transparent=True)

# stats for the above figure
states = ['unlabeled', 'theta', 'LIA']
num_b = 1000
g0 = events[m][measure][events[m][states[0]+'_bool']]
g1 = events[m][measure][events[m][states[1]+'_bool']]
g2 = events[m][measure][events[m][states[2]+'_bool']]
groups_list = [g0, g1, g2]
real_F, p_boot = boot_anova(groups_list, num_b)
# try the stats test again with a kruskal-wallace (nonparametric 1-way anova)
H, p_kw = stats.kruskal(g0, g1, g2, nan_policy='omit')


# do the pairwise t-tests
boot_t(g0, g1, 1000)
boot_t(g1, g2, 1000)
boot_t(g0, g2, 1000)
  
# some numbers from the histogram
l = 2
events[m][measure][events[m][states[l]+'_bool']].size
np.median(events[m][measure][events[m][states[l]+'_bool']])
np.std(events[m][measure][events[m][states[l]+'_bool']])
MADAM(events[m][measure][events[m][states[l]+'_bool']], np.median(events[m][measure][events[m][states[l]+'_bool']]))

