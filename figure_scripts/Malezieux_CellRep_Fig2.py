# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 14:28:38 2018

@author: Ashley
"""

# Manuscript Malezieux, Kees, Mulle submitted to Cell Reports
# Figure 2
# Description: brain state classification


# %% import modules


import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
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
def MADAM(data_pts, descriptor):
    v = np.sum(np.abs(data_pts-descriptor))/data_pts.size
    return v  

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
    
# def for finding which theta periods are associated with running
def find_run_theta(wh_ts, wh_speed, theta_start, theta_stop):
    run_theta = np.zeros(theta_start.size, dtype='bool')
    for i in np.arange(theta_start.size):
        if np.any(wh_speed[(wh_ts > theta_start[i]) &
           (wh_ts < theta_stop[i])] > 0):
            run_theta[i] = True
    run_theta_start = theta_start[run_theta]
    run_theta_stop = theta_stop[run_theta]
    nonrun_theta_start = theta_start[~run_theta]
    nonrun_theta_stop = theta_stop[~run_theta]
    return (run_theta_start, run_theta_stop, nonrun_theta_start,
            nonrun_theta_stop)    
    
# find timestamps of a state
def find_state_bool(ts, state_start, state_stop):
    state_bool = np.zeros(ts.size, dtype='bool')
    # find timestamps closest to running start and stop times
    for i in np.arange(state_start.size):
        ind_start = np.argmin(np.abs(ts-state_start[i]))
        ind_stop = np.argmin(np.abs(ts-state_stop[i]))
        state_bool[ind_start:ind_stop] = True
    return state_bool

# def for finding state times are associated with running
def find_run(wh_ts, wh_speed, state_ts):
    run_theta = np.zeros(state_ts.size, dtype='bool')
    for i in np.arange(state_ts.size):
        if np.any(wh_speed[(np.argmin(np.abs(wh_ts - state_ts[i])))] > 0):
            run_theta[i] = True
    run_state_ts = state_ts[run_theta]
    return (run_state_ts)

def find_lenght_run(state_run_ts):
    state_run_ts_df = pd.DataFrame(state_run_ts)
    state_run_ts_df2 = state_run_ts_df .set_index(state_run_ts)
    grouper = (~(pd.Series(state_run_ts_df2.index).diff() < 1)).cumsum().values  
    dfs = [dfx for _ , dfx in state_run_ts_df2.groupby(grouper)]
    jj = []
    for k in dfs:
        jj.append(len(k))
    jj = np.array(jj)
    return jj


def boot_pair_t(diff, num_b):
    real_d = np.mean(diff)
    faux_d = np.zeros(num_b)
    for b in np.arange(num_b):
        sample = np.random.choice([-1, 1], size = diff.size, replace=True)
        faux_d[b] = np.mean(diff*sample)
    p = np.sum(faux_d<real_d)/num_b
    return real_d, p
   
# %% Load data
    
dataset_folder = (r'C:\Users\akees\Documents\Ashley\Papers\MIND 1\Cell Reports\Dryad upload\Dataset')

cell_files = os.listdir(dataset_folder)
data = [{} for k in np.arange(len(cell_files))]
for i in np.arange(len(cell_files)):
    full_file = os.path.join(dataset_folder, cell_files[i])
    data[i] = np.load(full_file, allow_pickle=True).item()


# %% process data
    
# for each cell, find start and stop times for unlabeled times
for i in np.arange(len(data)):
    state_start = np.concatenate([data[i]['theta_start'], data[i]['LIA_start']])
    state_start = np.sort(state_start)
    state_stop = np.concatenate([data[i]['theta_stop'], data[i]['LIA_stop']])
    state_stop = np.sort(state_stop)
    data[i]['nost_start'] = np.append(data[i]['Vm_ds_ts'][0], state_stop)
    data[i]['nost_stop'] = np.append(state_start, data[i]['Vm_ds_ts'][-1])
 
#find nostate bool and nostate_ts
    nost_bool = find_state_bool(data[i]['Vm_ds_ts'], data[i]['nost_start'], data[i]['nost_stop'])
    data[i]['nost_bool'] = nost_bool
    data[i]['nost_ts'] = data[i]['Vm_ds_ts'][data[i]['nost_bool']]
    theta_bool = find_state_bool(data[i]['Vm_ds_ts'], data[i]['theta_start'], data[i]['theta_stop'])
    data[i]['theta_ts'] = data[i]['Vm_ds_ts'][theta_bool]

#find state times are associated with running
    run_state_ts = find_run(data[i]['wh_ts'], data[i]['wh_speed'], data[i]['nost_ts'])
    data[i]['nost_run'] = run_state_ts
    run_theta_ts = find_run(data[i]['wh_ts'], data[i]['wh_speed'], data[i]['theta_ts'])
    data[i]['theta_run'] = run_theta_ts
    
#find the lenght of each running bouts in timestamps during nostate and theta    
    jj = find_lenght_run(data[i]['nost_run'])
    data[i]['nost_run_len'] = jj
    kk = find_lenght_run(data[i]['theta_run'])
    data[i]['theta_run_len'] = kk

samp = (1/(data[0]['Vm_ds_ts'][1] - data[0]['Vm_ds_ts'][0]))    
nost_run_len_all = (np.concatenate(np.array([d['nost_run_len'] for d in data])))/samp
theta_run_len_all = (np.concatenate(np.array([d['theta_run_len'] for d in data])))/samp 
nost_run_ts_all = np.concatenate(np.array([d['nost_run'] for d in data]))
nost_ts_all = np.concatenate(np.array([d['nost_ts'] for d in data]))
#gives the % of nostate time spend running
nost_run_time_perc = (nost_run_ts_all.size*100)/nost_ts_all.size    

# change pupil to be expressed in xminimum, instead of pixles
for i in np.arange(len(data)):
    if data[i]['pupil'].size > 0:
        data[i]['norm_pupil'] = data[i]['pupil']/np.min(data[i]['pupil'])

 
states = [{'id': 'run_theta'}, {'id': 'nonrun_theta'}, {'id': 'LIA'}, {'id': 'nost'}]
for l in np.arange(len(states)):
    isi = np.empty(0)
    duration = np.empty(0)
    pupil = np.empty(0)
    norm_pupil = np.empty(0)
    dt = np.empty(0)
    for i in np.arange(len(data)):
        for j in np.arange(data[i][states[l]['id']+'_start'].size):
            duration = np.append(duration, (data[i][states[l]['id']+'_stop'][j] -
                                data[i][states[l]['id']+'_start'][j]))
            dt_bool = np.logical_and(data[i]['spec_ts']>data[i][states[l]['id']+'_start'][j],
                                           data[i]['spec_ts']<data[i][states[l]['id']+'_stop'][j])
            dt = np.append(dt, np.nanmean(data[i]['theta_delta'][dt_bool]))
            if data[i]['pupil'].size == 0:
                pupil = np.append(pupil, np.nan)
                norm_pupil = np.append(norm_pupil, np.nan)
            else:
                pupil_bool = np.logical_and(data[i]['pupil_ts']>data[i][states[l]['id']+'_start'][j],
                                           data[i]['pupil_ts']<data[i][states[l]['id']+'_stop'][j])
                pupil = np.append(pupil, np.nanmean(data[i]['pupil'][pupil_bool]))
                norm_pupil = np.append(norm_pupil, np.nanmean(data[i]['norm_pupil'][pupil_bool]))
            if j == 0:
                isi = np.append(isi, (data[i][states[l]['id']+'_start'][j] -
                                data[i]['Vm_ds_ts'][0]))
            else:
                isi = np.append(isi, (data[i][states[l]['id']+'_stop'][j] -
                                data[i][states[l]['id']+'_start'][j-1]))
    states[l]['isi'] = isi
    states[l]['duration'] = duration
    states[l]['pupil'] = pupil
    states[l]['norm_pupil'] = norm_pupil
    states[l]['theta_delta'] = dt


# take run theta triggered windows of running
t_win = [-2, 4]
for i in np.arange(len(data)):
    t_td, t_spec_ts = prepare_eta(data[i]['theta_delta'], data[i]['spec_ts'],
                                  data[i]['run_theta_start'], t_win)
    t_run, t_run_ts = prepare_eta(data[i]['wh_speed'], data[i]['wh_ts'],
                                  data[i]['run_theta_start'], t_win)
    data[i]['t_theta_delta'] = t_td
    data[i]['t_run'] = t_run

# put all triggered traces into one array
all_t_theta_delta = np.zeros([t_spec_ts.size, 0])
all_t_run = np.zeros([t_run_ts.size, 0])
for i in np.arange(len(data)):
    if data[i]['run_theta_start'].size > 0:
        all_t_theta_delta = np.append(all_t_theta_delta, data[i]['t_theta_delta'], axis=1)
        all_t_run = np.append(all_t_run, data[i]['t_run'], axis=1)
# remove nans
no_nan = np.logical_and((np.isnan(all_t_theta_delta[0, :])==0),
                        (np.isnan(all_t_run[0, :])==0))
all_t_theta_delta = all_t_theta_delta[:, no_nan]
all_t_run = all_t_run[:, no_nan]

# find the time lag between each run theta and run
for i in np.arange(len(data)):
    if data[i]['run_theta_start'].size == 0:
        data[i]['theta_run_lag'] = np.empty(0)
    else:
        if data[i]['run_start'].size == 0:
            data[i]['theta_run_lag'] = np.nan*np.ones(data[i]['run_theta_start'].size)
        else:
            lag = np.zeros(data[i]['run_theta_start'].size)
            for j in np.arange(data[i]['run_theta_start'].size):
                ind = np.argmin(np.abs(data[i]['run_start'] - data[i]['run_theta_start'][j]))
                lag[j] = data[i]['run_start'][ind] - data[i]['run_theta_start'][j]
            data[i]['theta_run_lag'] = lag

# put all run theta lags into one array
all_theta_run_lag = np.empty(0)
for i in np.arange(len(data)):
    if data[i]['run_theta_start'].size > 0:
        all_theta_run_lag = np.append(all_theta_run_lag, data[i]['theta_run_lag'])

# find the percent of time spent in each state
state = ['LIA', 'run_theta', 'nonrun_theta']
total_time = np.zeros(len(state))
grand_total = np.array(0)
for i in np.arange(len(data)):
    for l in np.arange(len(state)):
        time = np.sum(data[i][state[l]+'_stop'] - data[i][state[l]+'_start'])
        total_time[l] = total_time[l] + time
    grand_total = grand_total + (data[i]['Vm_ds_ts'][-1] - data[i]['Vm_ds_ts'][0])



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
fig_folder = r'C:\Users\akees\Documents\Ashley\Figures\2020-05_Paper_MIND1\Fig2'



#%% make figures - pie chart


# prep numbers for the pie chart
time_LIA = total_time[0]
time_run_theta = total_time[1]
time_nonrun_theta = total_time[2]
time_unlabeled = grand_total - sum(total_time)
colors = c_LIA, c_run_theta, c_nonrun_theta, c_wht
labels = 'LIA', 'run theta', 'rest theta', 'unlabeled'
pie_sizes = [time_LIA, time_run_theta, time_nonrun_theta, time_unlabeled]
# pie chart of hyp/dep/no change Vm
plt.figure(figsize=[1.2, 1.2])
plt.pie(pie_sizes, colors=colors)
plt.tight_layout()
plt.savefig(os.path.join(fig_folder, 'time_pie.png'), transparent=True)


time_LIA/grand_total
time_run_theta/grand_total
time_nonrun_theta/grand_total
time_unlabeled/grand_total

time_LIA
time_run_theta
time_nonrun_theta
time_unlabeled


# %% make figures - cumulative histograms

# make histograms of different state measurements
measure = 'theta_delta'

if measure == 'isi':
    bins = np.arange(0, 180, 0.5)
if measure == 'duration':
    bins = np.arange(0, 30, 0.01)
if measure == 'pupil':
    bins = np.arange(0, 50, 0.1)
if measure == 'norm_pupil':  # use this one
    bins = np.arange(1, 3, 0.001)
if measure == 'theta_delta':
    bins = np.arange(-2, 4, 0.01)
# create a figure with axes of defined size
fig = plt.figure(figsize=[1.6, 1.6])
# The first items are for padding and the second items are for the axes.
# sizes are in inch.
h = [Size.Fixed(0.5), Size.Fixed(0.9)]
v = [Size.Fixed(0.5), Size.Fixed(0.8)]
divider = Divider(fig, (0.0, 0.0, 1., 1.), h, v, aspect=False)
ax = Axes(fig, divider.get_position())
ax.set_axes_locator(divider.new_locator(nx=1, ny=1))
fig.add_axes(ax)
if measure == 'norm_pupil':
    ax.hist([states[0][measure][np.isnan(states[0][measure]) == 0],
              states[1][measure][np.isnan(states[1][measure]) == 0],
              states[2][measure][np.isnan(states[2][measure]) == 0],
              states[3][measure][np.isnan(states[3][measure]) == 0]],
             bins=bins, color=[c_grn, c_run_theta, c_bwn, c_mgry], density=True,
             label=['run theta', 'rest theta', 'LIA', 'nost'], histtype='step',
             cumulative=True)
else:
    ax.hist([states[0][measure][np.isnan(states[0][measure]) == 0],
              states[1][measure][np.isnan(states[1][measure]) == 0],
              states[2][measure][np.isnan(states[2][measure]) == 0]],
             bins=bins, color=[c_grn, c_run_theta, c_bwn], density=True,
             label=['run theta', 'rest theta', 'LIA'], histtype='step',
             cumulative=True)
if measure == 'isi':
    #ax.legend(frameon=False, loc=2, fontsize=20)
    ax.set_xlim([0, 120])
    ax.set_xticks([0, 60, 120]) 
    ax.set_xlabel('inter-event interval (s)')
if measure == 'duration':
    ax.set_xlim([0, 10])
    ax.set_xticks([0, 5, 10])
    ax.set_xlabel('event duration (s)')
if measure == 'pupil':
    ax.set_xlim([10, 40])
    ax.set_xticks([10, 20, 30, 40])
if measure == 'norm_pupil':
    ax.set_xlim([1, 2.5])
    ax.set_xticks([1, 1.5, 2, 2.5])
    ax.set_xlabel('pupil diameter\n(fold increase)')
if measure == 'theta_delta':
    ax.set_xlim([-1, 3])
    ax.set_xticks([-1, 0, 1, 2, 3])
    ax.set_xlabel('theta/delta power (z)')
ax.set_ylim([0, 1.1])
ax.set_yticks([0, 0.5, 1])
ax.set_yticklabels([0, '', 1])
ax.set_ylabel('cum. prop.')
ax.spines['left'].set_bounds(0, 1)
plt.savefig(os.path.join(fig_folder, measure + '.png'), transparent=True)


measure = 'theta_delta'
num_b = 1000
g0 = states[0][measure][np.isnan(states[0][measure]) == 0]
g1 = states[1][measure][np.isnan(states[1][measure]) == 0]
g2 = states[2][measure][np.isnan(states[2][measure]) == 0]
group_list = [g0, g1, g2]
if measure == 'norm_pupil':
    g3 = states[3][measure][np.isnan(states[3][measure]) == 0]
    group_list = [g0, g1, g2, g3]
boot_anova(group_list, num_b)
# try the stats test again with a kruskal-wallace (nonparametric 1-way anova)
H, p_kw = stats.kruskal(g0, g1, g2, nan_policy='omit')
H, p_kw = stats.kruskal(g0, g1, g2, g3, nan_policy='omit')

# do the pairwise t-tests
boot_t(g0, g1, 1000)
boot_t(g0, g2, 1000)
boot_t(g1, g2, 1000)
# for norm_pupil, compare all to g3 (nost)
boot_t(g3, g0, 1000)
boot_t(g3, g1, 1000)
boot_t(g3, g2, 1000)

# some numbers from the histogram
l = 2
states[l][measure][np.isnan(states[l][measure]) == 0].size
np.median(states[l][measure][np.isnan(states[l][measure]) == 0])
np.std(states[l][measure][np.isnan(states[l][measure]) == 0])
MADAM(states[l][measure][np.isnan(states[l][measure]) == 0], np.median(states[l][measure][np.isnan(states[l][measure]) == 0]))
           

# %% make figures - histogram of time lags


# plot the histogram of theta run lags
bins = np.arange(-6, 6.25, 0.25)
# create a figure with axes of defined size
fig = plt.figure(figsize=[1.6, 1.6])
# The first items are for padding and the second items are for the axes.
# sizes are in inch.
h = [Size.Fixed(0.5), Size.Fixed(0.9)]
v = [Size.Fixed(0.5), Size.Fixed(0.8)]
divider = Divider(fig, (0.0, 0.0, 1., 1.), h, v, aspect=False)
ax = Axes(fig, divider.get_position())
ax.set_axes_locator(divider.new_locator(nx=1, ny=1))
fig.add_axes(ax)
ax.hist(all_theta_run_lag[np.isnan(all_theta_run_lag) == 0], bins=bins,
        color=c_mgry, edgecolor='none')
ax.set_ylim([0, 44])
ax.set_yticks([0, 20, 40])
ax.set_xticks([-6, -3, 0, 3, 6])
ax.tick_params(top=False, right=False)
ax.spines['left'].set_bounds(0, 40)
ax.set_ylabel('num. of events')
ax.set_xlabel('time between running\nand theta onset (s)')
plt.savefig(os.path.join(fig_folder, 'run_lag.png'), transparent=True)


# some numbers from the histogram
all_theta_run_lag[np.isnan(all_theta_run_lag) == 0].size
np.nanmedian(all_theta_run_lag)
np.nanstd(all_theta_run_lag)
MADAM(all_theta_run_lag[np.isnan(all_theta_run_lag) == 0], np.nanmedian(all_theta_run_lag))
     

# bootstrap: test to see if lag is different than zero
num_b = 1000
faux_d = np.zeros(num_b)
real_d = np.nanmedian(all_theta_run_lag)
# change the dataset to have a mean of 0 (demean)
lag0 = all_theta_run_lag-real_d
for b in np.arange(num_b):
    f_select = np.random.randint(0, lag0.size, lag0.size)
    f_lag = lag0[f_select]
    faux_d[b] = np.nanmedian(f_lag)
p = np.sum(faux_d > real_d)/num_b



