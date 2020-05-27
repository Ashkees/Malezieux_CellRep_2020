# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 17:03:15 2019

@author: Ashley
"""


# Manuscript Malezieux, Kees, Mulle submitted to Cell Reports
# Figure S5 and S6 - spike threshold
# Description: changes in spike threshold with theta and LIA, plotted separately



# %% import modules

import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import rgb2hex
from pingouin import ancova
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

# find timestamps where running is nonzero
def find_running_bool(ts, run_start, run_stop):
    run_bool = np.zeros(ts.size, dtype='bool')
    # find timestamps closest to running start and stop times
    for i in np.arange(run_start.size):
        ind_start = np.argmin(np.abs(ts-run_start[i]))
        ind_stop = np.argmin(np.abs(ts-run_stop[i]))
        run_bool[ind_start:ind_stop] = True
    return run_bool

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
        et_waves_ind = [np.empty(0, dtype=int) for k in np.arange(event_times.size)]
    return et_signal, et_waves_ind


# %% load data

dataset_folder = (r'C:\Users\akees\Documents\Ashley\Papers\MIND 1\Cell Reports\Dryad upload\Dataset')

cell_files = os.listdir(dataset_folder)
data = [{} for k in np.arange(len(cell_files))]
for i in np.arange(len(cell_files)):
    full_file = os.path.join(dataset_folder, cell_files[i])
    data[i] = np.load(full_file, allow_pickle=True).item()



states = [{'id':'theta', 'bef':-2.5, 'aft':0.5, 'samp_time':2, 't_win':[-5, 5]},
          {'id':'LIA', 'bef':-4, 'aft':-1, 'samp_time':2, 't_win':[-5, 5]}]
ntl = ['nost', 'theta', 'LIA']
dVm_id = ['hyp', 'no', 'dep']

keep_cells = [isinstance(d['cell_id'], int) for d in data]
keep_cells_thresh = np.where([isinstance(d['cell_id'], int) for d in data])[0]


# %% process data


# for each cell, find start and stop times for unlabeled times
for i in np.arange(len(data)):
    state_start = np.concatenate([data[i]['theta_start'], data[i]['LIA_start']])
    state_start = np.sort(state_start)
    state_stop = np.concatenate([data[i]['theta_stop'], data[i]['LIA_stop']])
    state_stop = np.sort(state_stop)
    data[i]['nost_start'] = np.concatenate([np.array([0]), state_stop])
    data[i]['nost_stop'] = np.append(state_start, data[i]['Vm_ds_ts'][-1])


# for each cell, make a new spike_times for specifically non-spikelets
for i in np.arange(len(data)):
    data[i]['spike_times'] = np.delete(data[i]['sp_times'],
                                       data[i]['spikelets_ind'])


# for each cell, calculate the isi (inter-spike-interval)
# use sp_times - so, all spikes, including spikelets
for i in np.arange(len(data)):
    if data[i]['sp_times'].size > 0:
        isi0 = data[i]['sp_times'][0] - data[i]['Vm_ds_ts'][0]
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



# collect the Vm and threshold of spike in each state
sp_ntl = [{} for l in np.arange(len(ntl))]
for l in np.arange(len(ntl)):
    psd = np.empty(0)
    th_dist = np.empty(0)
    thresh = np.empty(0)
    prior_isi = np.empty(0)
    for k in np.arange(keep_cells_thresh.size):
        i = keep_cells_thresh[k]
        psd = np.append(psd, data[i]['psd'][data[i][ntl[l]+'_thresh_bool']])
        thresh = np.append(thresh, data[i]['thresh_Vm'][data[i][ntl[l]+'_thresh_bool']])
        th_dist = np.append(th_dist, data[i]['th_dist'][data[i][ntl[l]+'_thresh_bool']])
        prior_isi = np.append(prior_isi, data[i]['isi'][data[i]['th_ind']][data[i][ntl[l]+'_thresh_bool']])
    sp_ntl[l]['psd'] = psd
    sp_ntl[l]['thresh'] = thresh
    sp_ntl[l]['th_dist'] = th_dist
    sp_ntl[l]['prior_isi'] = prior_isi
    sp_ntl[l]['pre_sp_Vm'] = thresh + th_dist



# determine which cells are hyp/dep/no
for l in np.arange(len(states)):
    cell_p = np.array([d[states[l]['id']+'_cell_p'] for d in data])
    cell_dVm = np.zeros(cell_p.size)
    cell_dVm[cell_p < 0.05] = -1
    cell_dVm[cell_p > 0.95] = 1
    states[l]['cell_dVm'] = cell_dVm

    


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
c_state = [c_mgry, c_run_theta, c_lbwn]
c_state_dark = [c_dgry, c_grn, c_bwn]
c_tl = [c_run_theta, c_lbwn]
c_tnl = [c_run_theta, c_blk, c_lbwn]
#c_sp_type = ['#440154', '#287D8E', '#29AF7F', '#95D840']
c_sp_type = ['#55C667', '#238A8D', '#404788', rgb2hex(c_blk)]


viridis_20 = ['#440154', '#481567', '#482677', '#453781', '#404788',
              '#39568C', '#33638D', '#2D708E', '#287D8E', '#238A8D',
              '#1F968B', '#20A387', '#29AF7F', '#3CBB75', '#55C667',
              '#73D055', '#95D840', '#B8DE29', '#DCE319', '#FDE725'] 
viridis_6 = ['#440154', '#453781', '#33638D', '#238A8D', '#29AF7F', '#73D055']
viridis_5 = ['#440154', '#404788', '#287D8E', '#29AF7F', '#95D840'] 


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



## theta only
## set which states to plot
#d_l = [0, 1]
## set figure output folder
#fig_folder = r'C:\Users\akees\Documents\Ashley\Figures\2020-05_Paper_MIND1\FigS5'


# LIA only
# set which states to plot
d_l = [0, 2]
# set figure output folder
fig_folder = r'C:\Users\akees\Documents\Ashley\Figures\2020-05_Paper_MIND1\FigS6'




# %% panels for the actual figure

keep_cells = [isinstance(d['cell_id'], int) for d in data]
theta_cell_p = np.array([d['theta_cell_p'] for d in data])[keep_cells]
LIA_cell_p = np.array([d['LIA_cell_p'] for d in data])[keep_cells]

c_state_hist = [c_mgry, c_grn, c_bwn]
c_state_scat = [c_lgry, c_run_theta, c_lbwn]


# relationship between Vm and absolute threshold
# VERSION: trend lines for each state
# Version: theta and LIA plotted separately
m = np.full(len(ntl), np.nan)
b = np.full(len(ntl), np.nan)
r = np.full(len(ntl), np.nan)
l_p = np.full(len(ntl), np.nan)
stderr = np.full(len(ntl), np.nan)
n = np.full(len(ntl), np.nan)
lim1 = -55
lim2 = -15
size = 2
fig, ax = plt.subplots(2, 2, sharex='col', sharey='row', figsize=[3.5, 3.5],
                       gridspec_kw = {'height_ratios':[1, 2], 'width_ratios':[2, 1]})
# nothing in the top right corner
ax[0, 1].axis('off')
# scatter of absolute threshold vs Vm in bottom left
for l in d_l:
    x = sp_ntl[l]['pre_sp_Vm']
    y = sp_ntl[l]['thresh']
    # remove spikes when pre_spike_Vm is 0 (due to nans when preparing the eta)
    inds = x != 0
    x = x[inds]
    y = y[inds]
    # remove outliers
    inds = np.abs(stats.zscore(y/x))<4
    x = x[inds]
    y = y[inds]
    ax[1,0].scatter(x, y, s=size,
                    color=c_state_scat[l], zorder=2, alpha=0.5)
    # put in the state-specific linear regression
    m[l], b[l], r[l], l_p[l], stderr[l] = stats.linregress(x, y)
    fit_line = m[l]*x+b[l]
    ax[1,0].plot(x, fit_line, color=c_state_hist[l], zorder=3)
    n[l] = x.size
ax[1,0].plot([lim1, lim2], [lim1, lim2], color=c_blk, linestyle='--', zorder=1)
ax[1,0].set_xlim([-56, -20])
ax[1,0].set_ylim([-50, -14])
ax[1,0].set_xticks([-50, -40, -30, -20]) 
ax[1,0].set_xticklabels([-50, -40, -30, -20], fontsize=8) 
ax[1,0].set_xlabel('Vm 50-300 ms before the spike (mV)', fontsize=8)
ax[1,0].set_yticks([-50, -40, -30, -20])
ax[1,0].set_yticklabels([-50, -40, -30, -20], fontsize=8)  
ax[1,0].set_ylabel('absolute threshold (mV)', fontsize=8)
# histogram of absolute threshold (averaged over cells with CI) on right
bins = np.arange(-60, 0, 2)
H = np.full([len(data), bins.size-1, len(ntl)], np.nan)
for i in np.arange(len(data)):
    for l in np.arange(len(ntl)):
        H[i, :, l] = np.histogram(data[i]['thresh_Vm'][data[i][ntl[l]+'_thresh_bool']],
                                  bins=bins)[0]
        # normalize to total number of spikes
        H[i, :, l] = H[i, :, l]/np.sum(H[i, :, l])
# remove extra recordings from cells
H = H[keep_cells_thresh, :, :]
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
# plot the mean hist
for l in d_l:
    ax[1, 1].plot(H_mean[:, l], bins[:-1], color=c_state_hist[l], zorder=2)
    ax[1, 1].fill_betweenx(bins[:-1], H_CI_low[:, l], H_CI_high[:, l],
                    facecolor=c_state_scat[l], linewidth=0, zorder=1, alpha=0.25)
ax[1, 1].set_xticks([0, 0.1, 0.2, 0.3]) 
ax[1, 1].set_xticklabels([0, '', '', 0.3], fontsize=8)
ax[1, 1].set_xlabel('proportion of spikes', fontsize=8)
# histogram of base Vm (averaged over cells with CI) on top
bins = np.arange(-60, 0, 2)
H = np.full([len(data), bins.size-1, len(ntl)], np.nan)
for i in np.arange(len(data)):
    for l in np.arange(len(ntl)):
        H[i, :, l] = np.histogram(data[i]['pre_spike_Vm'][data[i][ntl[l]+'_thresh_bool']],
                                  bins=bins)[0]
        # normalize to total number of spikes
        H[i, :, l] = H[i, :, l]/np.sum(H[i, :, l])
# remove extra recordings from cells
H = H[keep_cells_thresh, :, :]
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
# plot the mean hist
for l in d_l:
    ax[0, 0].plot(bins[:-1], H_mean[:, l], color=c_state_hist[l], zorder=2)
    ax[0, 0].fill_between(bins[:-1], H_CI_low[:, l], H_CI_high[:, l],
                    facecolor=c_state_scat[l], linewidth=0, zorder=1, alpha=0.25)
ax[0, 0].set_yticks([0, 0.1, 0.2, 0.3]) 
ax[0, 0].set_yticklabels([0, '', '', 0.3], fontsize=8)
ax[0, 0].set_ylabel('proportion of spikes', fontsize=8)
fig.tight_layout()
plt.savefig(os.path.join(fig_folder, ntl[d_l[-1]]+'_scat_absolute_threshold.png'), transparent=True)


# ANCOVA for the scatterplot above:
# factors = nostate, theta, LIA
# depedent variable = absolute spike threshold
# covarying variables = pre spike Vm, prior isi
# put information of sp_ntl into dataframe, then concatenate
for l in np.arange(len(ntl)):
    state_df = pd.DataFrame(columns=['absolute_threshold', 'relative_threshold',
                           'pre_spike_Vm', 'prior_isi', 'state'])
    # remove bad values and outliers
    x = sp_ntl[l]['pre_sp_Vm']
    y = sp_ntl[l]['thresh']
    inds = np.logical_and(x != 0, np.abs(stats.zscore(x/y))<4)
    state_df['absolute_threshold'] = sp_ntl[l]['thresh'][inds]
    state_df['relative_threshold'] = -1*sp_ntl[l]['th_dist'][inds]
    state_df['pre_spike_Vm'] = sp_ntl[l]['pre_sp_Vm'][inds]
    state_df['prior_isi'] = sp_ntl[l]['prior_isi'][inds]
    state_df['state'] = np.full(sp_ntl[l]['thresh'][inds].size, ntl[l])
    sp_ntl[l]['state_df'] = state_df
df = pd.concat([d['state_df'] for d in sp_ntl])
#df = pd.concat([sp_ntl[0]['state_df'], sp_ntl[2]['state_df']])
# try the ancova using pingouin
#ancova_results = ancova(data=df, dv='absolute_threshold',
#                        covar=['pre_spike_Vm', 'prior_isi'], between='state')
ancova_results = ancova(data=df, dv='absolute_threshold',
                        covar='pre_spike_Vm', between='state')
## test whether the slopes are different - don't know how to interpret??
#lm = ols(formula = 'absolute_threshold ~ pre_spike_Vm * state', data = df)
#fit = lm.fit()
#fit.summary()



# %%

# stats for pre_spike_Vm vs absolute_threshold over cells instead of over spikes
S_slope = np.full([len(data), len(ntl)], np.nan)
S_intcpt = np.full([len(data), len(ntl)], np.nan)
n = np.full([len(data), len(ntl)], np.nan)

# do the linear regression for each cell and each state
for i in np.arange(len(data)):
    for l in np.arange(len(ntl)):
        thresh = data[i]['thresh_Vm'][data[i][ntl[l]+'_thresh_bool']]
        th_dist = data[i]['th_dist'][data[i][ntl[l]+'_thresh_bool']]
        x = th_dist + thresh
        y = thresh
        # remove spikes when pre_spike_Vm is 0 (due to nans when preparing the eta)
        inds = x != 0
        x = x[inds]
        y = y[inds]
        # remove outliers
        inds = np.abs(stats.zscore(y/x))<4
        x = x[inds]
        y = y[inds]
        n[i, l] = x.size
        if x.size > 0:
            # put in the state-specific linear regression
            m, b, r, l_p, stderr = stats.linregress(x, y)
            S_slope[i, l] = m
            S_intcpt[i, l] = b
# remove extra recordings from cells        
S_slope = S_slope[keep_cells, :]
S_intcpt = S_intcpt[keep_cells, :]
n = n[keep_cells, :]

# do paired stats on nost vs theta and nost vs LIA - test if slopes are changing
num_b = 1000
p = np.full(len(ntl) - 1, np.nan)
d = np.full(len(ntl) - 1, np.nan)
for l in np.arange(len(ntl) - 1):
    dif = S_slope[:, l+1] - S_slope[:, 0]
#    # remove nans
#    dif = dif[~np.isnan(dif)]
    # remove nans and when there is a negative slope
    dif = dif[S_slope[:, l+1] > 0]
    d[l], p[l] = boot_pair_t(dif, num_b)
    print(dif.size)
print(d)
print(p)

# do paired stats on nost vs theta and nost vs LIA - test if intercepts are changing
num_b = 1000
p = np.full(len(ntl) - 1, np.nan)
d = np.full(len(ntl) - 1, np.nan)
for l in np.arange(len(ntl) - 1):
    dif = S_intcpt[:, l+1] - S_intcpt[:, 0]
#    # remove nans
#    dif = dif[~np.isnan(dif)]
    # remove nans and when there is a negative slope - assume something went wrong
    dif = dif[S_slope[:, l+1] > 0]
    d[l], p[l] = boot_pair_t(dif, num_b)
    print(dif.size)
print(d)
print(p)        


# %%

# do the stats for the above figure - pre_spike_Vm
S = np.full([len(data), len(ntl)], np.nan)
for i in np.arange(len(data)):
    for l in np.arange(len(ntl)):
        pre_spike_Vm = data[i]['pre_spike_Vm'][data[i][ntl[l]+'_thresh_bool']]
        if pre_spike_Vm.size > 0:
            S[i, l] = np.nanmedian(pre_spike_Vm)
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
# LIA nondep cells
dif = S[:, 2][LIA_cell_p < 0.95] - S[:, 0][LIA_cell_p < 0.95]
# remove nans
dif = dif[~np.isnan(dif)]
d, p = boot_pair_t(dif, num_b)
print(dif.size)
print(d)
print(p)

# descriptive numbers
l = 2
np.sum(~np.isnan(S[:, l]))
np.nanmedian(S[:, l])
np.nanstd(S[:, l])
MADAM(S[:, l], np.nanmedian(S[:, l]))



# do the stats for the above figure - absolute threshold
S = np.full([len(data), len(ntl)], np.nan)
for i in np.arange(len(data)):
    for l in np.arange(len(ntl)):
        pre_spike_Vm = data[i]['thresh_Vm'][data[i][ntl[l]+'_thresh_bool']]
        if pre_spike_Vm.size > 0:
            S[i, l] = np.nanmedian(pre_spike_Vm)
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
# LIA nondep cells
dif = S[:, 2][LIA_cell_p < 0.95] - S[:, 0][LIA_cell_p < 0.95]
# remove nans
dif = dif[~np.isnan(dif)]
d, p = boot_pair_t(dif, num_b)
print(dif.size)
print(d)
print(p)

# descriptive numbers
l = 2
np.sum(~np.isnan(S[:, l]))
np.nanmedian(S[:, l])
np.nanstd(S[:, l])
MADAM(S[:, l], np.nanmedian(S[:, l]))


# do the spike-based stats for the figure above
measure = 'pre_sp_Vm'
measure = 'thresh'

num_b = 1000
g0 = sp_ntl[0][measure]
g1 = sp_ntl[1][measure]
g2 = sp_ntl[2][measure]
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
l = 2
sp_ntl[l][measure].size
np.nanmedian(sp_ntl[l][measure])
np.nanstd(sp_ntl[l][measure])
MADAM(sp_ntl[l][measure], np.nanmedian(sp_ntl[l][measure]))





# %% paired scatters and box plots



# paired scatter of relative threshold between different states
# VERSION: cells/lines colored according to hyp/dep/no
# version: theta and LIA plotted separately
# set up the numbers
measure = np.full([len(data), len(ntl)], np.nan)
for i in np.arange(len(data)):
    for l in np.arange(len(ntl)):
        # remove outliers
        values = data[i]['th_dist'][data[i][ntl[l]+'_thresh_bool']]
        values = values[np.logical_and(values > -15, values < 0)]
        measure[i, l] = -1*np.nanmedian(values)
# remove unwanted cells
measure = measure[keep_cells_thresh, :]
# make the plot
#fig, ax = plt.subplots(1, figsize=[1.5, 1.75])
# create a figure with axes of defined size
fig = plt.figure(figsize=[2, 2])
# The first items are for padding and the second items are for the axes.
# sizes are in inch.
h = [Size.Fixed(0.5), Size.Fixed(1.1)]
v = [Size.Fixed(0.5), Size.Fixed(1.2)]
divider = Divider(fig, (0.0, 0.0, 1., 1.), h, v, aspect=False)
ax = Axes(fig, divider.get_position())
ax.set_axes_locator(divider.new_locator(nx=1, ny=1))
fig.add_axes(ax)
line_x = np.array([1.75, 3.25])
bar_x = np.array([1, 4])
y = measure[:, d_l]
dVm = states[d_l[-1]-1]['cell_dVm'][keep_cells_thresh]
for i in np.arange(y.shape[0]):
    ax.plot(line_x, y[i, :], color=c_lgry, zorder=1)
    if dVm[i] == -1:
        ax.plot(line_x, y[i, :], color=c_hyp, zorder=2)
    elif dVm[i] == 1:
        ax.plot(line_x, y[i, :], color=c_dep, zorder=2)
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
ax.set_xticklabels(['unlabeled', ntl[d_l[1]]], fontsize=8)
ax.set_ylim([0, 13])
ax.set_yticks([0, 4, 8, 12])
ax.set_yticklabels([0, 4, 8, 12], fontsize=8)
ax.set_ylabel('relative threshold (mV)', fontsize=8)
ax.set_xlim([0, bar_x[1]+1])
ax.spines['bottom'].set_visible(False)
plt.savefig(os.path.join(fig_folder, ntl[d_l[-1]]+'_pair_relative_threshold.png'), transparent=True)




# do the stats for the above plot
# paired t-test for theta vs no state
diff = measure[:,1] - measure[:,0]
diff = diff[~np.isnan(diff)]
real_d, p = boot_pair_t(diff, num_b)
# paired t-test for LIA vs no state
diff = measure[:,2] - measure[:,0]
diff = diff[~np.isnan(diff)]
real_d, p = boot_pair_t(diff, num_b)

# do the paired boot stats for theta hyp and LIA dep cells only
num_b = 1000
# theta hyp cells
dif = measure[:, 1][theta_cell_p < 0.05] - measure[:, 0][theta_cell_p < 0.05]
# remove nans
dif = dif[~np.isnan(dif)]
d, p = boot_pair_t(dif, num_b)
print(dif.size)
print(d)
print(p)
# LIA dep cells
dif = measure[:, 2][LIA_cell_p > 0.95] - measure[:, 0][LIA_cell_p > 0.95]
# remove nans
dif = dif[~np.isnan(dif)]
d, p = boot_pair_t(dif, num_b)
print(dif.size)
print(d)
print(p)

# descriptive numbers
l = 2
np.sum(~np.isnan(measure[:, l]))
np.nanmedian(measure[:, l])
np.nanstd(measure[:, l])
MADAM(measure[:, l], np.nanmedian(measure[:, l]))


# find which cells have a significant change - nonparametric stats
p_kw = np.full(len(data), np.nan)
p_mw = np.full([len(data), len(states)], np.nan)
for i in np.arange(len(data)):
    groups_list = [None]*len(ntl)
    for l in np.arange(len(ntl)):
        # remove outliers
        values = data[i]['th_dist'][data[i][ntl[l]+'_thresh_bool']]
        values = values[np.logical_and(values > -15, values < 0)]
        groups_list[l] = values
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




#%% make figures for psd

# paired scatter of psd between different states
# VERSION: cells/lines colored according to hyp/dep/no
# version: theta and LIA plotted separately
# set up the numbers
measure = np.full([len(data), len(ntl)], np.nan)
for i in np.arange(len(data)):
    for l in np.arange(len(ntl)):
        # remove outliers
        values = data[i]['psd'][data[i][ntl[l]+'_thresh_bool']]
        values = values[np.logical_and(values > -10, values < 0)]
        measure[i, l] = -1*np.nanmedian(values)
# remove unwanted cells
measure = measure[keep_cells_thresh, :]
# make the plot
#fig, ax = plt.subplots(1, figsize=[1.5, 1.75])
# create a figure with axes of defined size
fig = plt.figure(figsize=[2, 2])
# The first items are for padding and the second items are for the axes.
# sizes are in inch.
h = [Size.Fixed(0.5), Size.Fixed(1.1)]
v = [Size.Fixed(0.5), Size.Fixed(1.2)]
divider = Divider(fig, (0.0, 0.0, 1., 1.), h, v, aspect=False)
ax = Axes(fig, divider.get_position())
ax.set_axes_locator(divider.new_locator(nx=1, ny=1))
fig.add_axes(ax)
line_x = np.array([1.75, 3.25])
bar_x = np.array([1, 4])
y = measure[:, d_l]
dVm = states[d_l[-1]-1]['cell_dVm'][keep_cells_thresh]
for i in np.arange(y.shape[0]):
    ax.plot(line_x, y[i, :], color=c_lgry, zorder=1)
    if dVm[i] == -1:
        ax.plot(line_x, y[i, :], color=c_hyp, zorder=2)
    elif dVm[i] == 1:
        ax.plot(line_x, y[i, :], color=c_dep, zorder=2)
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
ax.set_xticklabels(['unlabeled', ntl[d_l[1]]], fontsize=8)
ax.set_ylim([1, 8])
ax.set_yticks([2, 4, 6, 8])
ax.set_yticklabels([2, 4, 6, 8], fontsize=8)
ax.set_ylabel('pre-spike depolarization (mV)', fontsize=8)
ax.set_xlim([0, bar_x[1]+1])
ax.set_xlim([0, bar_x[1]+1])
ax.spines['bottom'].set_visible(False)
plt.savefig(os.path.join(fig_folder, ntl[d_l[-1]]+'_pair_psd.png'), transparent=True)


# do the stats for the above plot
# paired t-test for theta vs no state
diff = measure[:,1] - measure[:,0]
diff = diff[~np.isnan(diff)]
real_d, p = boot_pair_t(diff, num_b)
# paired t-test for LIA vs no state
diff = measure[:,2] - measure[:,0]
diff = diff[~np.isnan(diff)]
real_d, p = boot_pair_t(diff, num_b)

# do the paired boot stats for theta hyp and LIA dep cells only
num_b = 1000
# theta hyp cells
dif = measure[:, 1][theta_cell_p < 0.05] - measure[:, 0][theta_cell_p < 0.05]
# remove nans
dif = dif[~np.isnan(dif)]
d, p = boot_pair_t(dif, num_b)
print(dif.size)
print(d)
print(p)
# LIA dep cells
dif = measure[:, 2][LIA_cell_p > 0.95] - measure[:, 0][LIA_cell_p > 0.95]
# remove nans
dif = dif[~np.isnan(dif)]
d, p = boot_pair_t(dif, num_b)
print(dif.size)
print(d)
print(p)

# descriptive numbers
l = 2
np.sum(~np.isnan(measure[:, l]))
np.nanmedian(measure[:, l])
np.nanstd(measure[:, l])
MADAM(measure[:, l], np.nanmedian(measure[:, l]))


# find which cells have a significant change - nonparametric stats
p_kw = np.full(len(data), np.nan)
p_mw = np.full([len(data), len(states)], np.nan)
for i in np.arange(len(data)):
    groups_list = [None]*len(ntl)
    for l in np.arange(len(ntl)):
        # remove outliers
        values = data[i]['psd'][data[i][ntl[l]+'_thresh_bool']]
        values = values[np.logical_and(values > -15, values < 0)]
        groups_list[l] = values
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



# %% example traces for actual figure

# Vm threshold over time with Vm_s underneath


i = 31  # cell 90
fig, ax = plt.subplots(1, figsize=[3.5, 1.3])
ax.plot(data[i]['Vm_ds_ts'], data[i]['Vm_s_ds'], color=c_mgry, zorder=2)
ax.scatter(data[i]['thresh_times'], data[i]['thresh_Vm'], color=c_sp, s=2, zorder=2)
for j in np.arange(data[i]['theta_start'].size):
    ax.axvspan(data[i]['theta_start'][j], data[i]['theta_stop'][j],
                  ymin=0.05, ymax=0.8, color=c_run_theta, alpha=0.5, zorder=1)
for j in np.arange(data[i]['LIA_start'].size):
    ax.axvspan(data[i]['LIA_start'][j], data[i]['LIA_stop'][j],
                  ymin=0.05, ymax=0.8, color=c_LIA, alpha=0.5, zorder=1)
ax.set_ylim([-55, -27])
ax.set_xlim([240, 300])
ax.spines['bottom'].set_bounds(290, 300)
ax.set_xticks([295])
ax.xaxis.set_tick_params(length=0)
ax.set_xticklabels(['10 s'], fontsize=8)
ax.spines['left'].set_bounds(-55, -30)
ax.set_yticks([-50, -40, -30])
ax.set_yticklabels([-50, -40, -30], fontsize=8)
ax.set_ylabel('mV', rotation=0, verticalalignment='center', fontsize=8)
ax.yaxis.set_label_coords(-0.16, 0.53, transform=None)
ax.set_title('Cell 25', loc='left', fontsize=8)
ax.text(240, -29, 'spike thresholds', color=c_sp)
ax.text(240, -59, 'smoothed Vm', color=c_mgry)
fig.tight_layout()
plt.savefig(os.path.join(fig_folder, 'ex_thresh_time.png'), transparent=True)

