# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 15:05:36 2019

@author: Ashley
"""

# Manuscript Malezieux, Kees, Mulle submitted to Cell Reports
# Supplemental figure - Spikelets
# Description: quantitative aspects, firing rate of spikelets during theta/LIA

# %% import modules

import os
import numpy as np
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
            # deal into faux groups, each one the same size as in real data
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




#%% concatenate all the cell dataframes into one big dataframe - spikelets

df = pd.concat([d['spikelet_df'] for d in data])

# keep only hand-picked cells with true spikelets
spikelet_cells = [10, 12, 16, 50, 52, 56, 61, 67, 68, 88, 93, 100]
df = df[df['cell_id'].isin(spikelet_cells)]
# remove spikelets that have nan in max_rise
df = df[~np.isnan(df['max_rise'])]
# remove spikelets that have nan in decay_tau
df = df[~np.isnan(df['decay_tau'])]
# remove spikelets whose max_rise is greater than 50 (they're actually spikes)
df = df[df['max_rise'] < 50]



#%% concatenate all the cell dataframes into one big dataframe - spikes used in threshold analysis

th_df = pd.concat([d['th_df'] for d in data])
## keep only hand-picked cells with true spikelets
#th_df = th_df[th_df['cell_id'].isin(spikelet_cells)]



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
# spikelets
c_small = [0.957, 0.742, 0.254]  # yellow
#c_big = [0.75, 0.344, 0.02]  # orange
#c_small = [0.867, 0.477, 0.133]
#c_small = [0.922, 0.586, 0.641]  # pink
c_big = [0.027, 0.34, 0.355]  # dark teal
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

c_state_hist = [c_mgry, c_grn, c_bwn]
c_state_fill = [c_lgry, c_run_theta, c_lbwn]
c_spikelets = [c_big, c_small]

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
fig_folder = r'C:\Users\akees\Documents\Ashley\Figures\2020-05_Paper_MIND1\FigS4'


#%% make figures - cell paired boxplots for spike vs spikelet properties

g0_cells = [12, 16]
g1_cells = [10, 50, 52, 56, 61, 67, 68, 88, 93, 100]
cond0 = df['cell_id'].isin(g0_cells)
cond1 = df['cell_id'].isin(g1_cells)
c_splet = [c_small, c_sp]
all_cells = np.unique(th_df['cell_id'])

# fwhm
# thresh_Vm       
# amplitude      
# rise_time       
# decay_tau
# max_rise



#measure = 'amplitude'
measure = 'rise_time'
S = np.full([len(all_cells), 2], np.nan)
for i in np.arange(len(all_cells)):
    values = df[measure][(df['cell_id'] == all_cells[i])]
    S[i, 0] = np.nanmean(values)
    values = th_df[measure][(th_df['cell_id'] == all_cells[i])]
    S[i, 1] = np.nanmean(values)
      
#fig, ax = plt.subplots(1, figsize=[1.5, 1.5])
# create a figure with axes of defined size
fig = plt.figure(figsize=[2, 2])
# The first items are for padding and the second items are for the axes.
# sizes are in inch.
h = [Size.Fixed(0.5), Size.Fixed(1.1)]
v = [Size.Fixed(0.5), Size.Fixed(1.3)]
divider = Divider(fig, (0.0, 0.0, 1., 1.), h, v, aspect=False)
ax = Axes(fig, divider.get_position())
ax.set_axes_locator(divider.new_locator(nx=1, ny=1))
fig.add_axes(ax)
bar_x = np.array([1, 4])
line_x = np.array([0.75, 2.25])
for i in np.arange(S.shape[0]):
    for l in np.arange(S.shape[1]-1):
        ax.plot(bar_x[l]+line_x, S[i, l:l+2], color=c_small, zorder=1)
    if measure == 'amplitude':
        if S[i, 0] > 5:
            ax.plot(bar_x[l]+line_x, S[i, l:l+2], color=c_big, zorder=2)
        ax.set_yticks([0, 10, 20, 30, 40, 50, 60])
        ax.set_yticklabels([0, '', 20, '', 40, '', 60])
        ax.set_ylim([0, 60])
        ax.set_ylabel('amplitude (mV)')
    if measure == 'rise_time':
        if S[i, 0] > 0.5:
            ax.plot(bar_x[l]+line_x, S[i, l:l+2], color=c_big, zorder=2)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1, 1.25])
        ax.set_yticklabels([0, '', 0.5, '', 1, ''])
        ax.set_ylabel('rise time (ms)')
    if measure == 'max_rise':
        if S[i, 0] > 20:
            ax.plot(bar_x[l]+line_x, S[i, l:l+2], color=c_big, zorder=2)
for l in np.arange(S.shape[1]):
    no_nan = S[:, l]
    if l == 0:
        no_nan = no_nan[np.isin(all_cells, g1_cells)]
    if l == 1:
        no_nan = no_nan[~np.isnan(no_nan)]
    bp = ax.boxplot(no_nan, sym='', patch_artist=True, showfliers=True,
                     whis=[5, 95], widths=0.75, positions=[bar_x[l]])     
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=c_splet[l], linewidth=1.5)
    for patch in bp['boxes']:
        patch.set(facecolor=c_wht)
# add the flier dots for the big spikelet cells
fliers = S[:, 0][np.isin(all_cells, g0_cells)]
plt.scatter([bar_x[0], bar_x[0]], fliers, s=10, color=c_big)
ax.set_xticks(bar_x)
ax.xaxis.set_tick_params(length=0)
ax.set_xticklabels(['spikelets', 'spikes'])
ax.set_xlim([0, bar_x[-1]+1])
ax.spines['bottom'].set_visible(False)
plt.savefig(os.path.join(fig_folder, measure+'_spike_vs_spikelet.png'), transparent=True) 


# stats for above plot
# do the paired boot stats
base = 1  # index of spikes
comp = [0]  # index of spikelets
num_b = 1000
p = np.full(len(comp), np.nan)
d = np.full(len(comp), np.nan)
for l in np.arange(len(comp)):
    dif = S[:, comp[l]] - S[:, base]
    # remove nans
    dif = dif[~np.isnan(dif)]
    d[l], p[l] = boot_pair_t(dif, num_b)
    print(dif.size)
print(d)
print(p)

# some descriptive numbers (omit big spikelet cells)
S_nobig = S[~np.isin(all_cells, g0_cells), :]
l = 1
np.nanmedian(S_nobig[:, l])
np.nanstd(S_nobig[:, l])
MADAM(S_nobig[:, l], np.nanmedian(S_nobig[:, l]))


#%% make figures - cell paired boxplots for spikelet rate across 3 states

c_3state = [c_LIA, c_mgry, c_run_theta]
all_cell_id = [d['cell_id'] for d in data]
keep_cells = [isinstance(d['cell_id'], int) for d in data]
cell_id = np.array(all_cell_id)[keep_cells]
sec_rec = np.array(all_cell_id)[~np.array(keep_cells)]
theta_cell_p = np.array([d['theta_cell_p'] for d in data])[keep_cells]
LIA_cell_p = np.array([d['LIA_cell_p'] for d in data])[keep_cells]


# spikelet rate
test = ['LIA', 'nost', 'theta'] # this order must not change
S = np.full([len(spikelet_cells), len(test)], np.nan)
S_theta_p = np.full(len(spikelet_cells), np.nan)
S_LIA_p = np.full(len(spikelet_cells), np.nan)
for c in np.arange(len(spikelet_cells)):
    total_time = np.full(len(test), np.nan)
    total_spikelets = np.full(len(test), np.nan)
    # find the index of that spike's data
    i = all_cell_id.index(spikelet_cells[c])
    # record the theta and LIA cell_p
    S_theta_p[c] = data[i]['theta_cell_p']
    S_LIA_p[c] = data[i]['LIA_cell_p']
    # calculate total # spikelets per state
    total_spikelets[0] = sum((df['cell_id'] == spikelet_cells[c]) & (df['state'] == 'LIA'))
    total_spikelets[1] = sum((df['cell_id'] == spikelet_cells[c]) & (df['state'] == 'nost'))
    total_spikelets[2] = sum((df['cell_id'] == spikelet_cells[c]) & (df['state'] == 'theta'))
    # add up total time in the state
    for l in np.arange(len(test)):
        total_time[l] = np.sum(data[i][test[l]+'_stop'] - data[i][test[l]+'_start'])
    # find any secondary recordings to add time in states
    x = 1
    while (str(spikelet_cells[c])+'_'+str(x)) in all_cell_id:
        j = all_cell_id.index(str(spikelet_cells[c])+'_'+str(x))
        for l in np.arange(len(test)):
            total_time[l] = total_time[l] + np.sum(data[j][test[l]+'_stop'] - data[j][test[l]+'_start'])
        x = x+1
    S[c, :] = total_spikelets/total_time



# make the figure  - VERSION no pair lines     
#fig, ax = plt.subplots(1, figsize=[1.5, 1.5])
# create a figure with axes of defined size
fig = plt.figure(figsize=[2, 2])
# The first items are for padding and the second items are for the axes.
# sizes are in inch.
h = [Size.Fixed(0.5), Size.Fixed(1.1)]
v = [Size.Fixed(0.5), Size.Fixed(1.3)]
divider = Divider(fig, (0.0, 0.0, 1., 1.), h, v, aspect=False)
ax = Axes(fig, divider.get_position())
ax.set_axes_locator(divider.new_locator(nx=1, ny=1))
fig.add_axes(ax)
bar_x = np.array([1, 4, 7])
line_x = np.array([0.75, 2.25])
for l in np.arange(S.shape[1]):
    no_nan = S[:, l]
    no_nan = no_nan[~np.isnan(no_nan)]
    bp = ax.boxplot(no_nan, sym='', patch_artist=True,
                         whis=[5, 95], widths=2, positions=[bar_x[l]])     
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=c_3state[l], linewidth=1.5)
    for patch in bp['boxes']:
        patch.set(facecolor=c_wht)
ax.set_xticks([0, 4, 8])
ax.xaxis.set_tick_params(length=0)
ax.set_xticklabels(['LIA', 'unlabeled', 'theta'])
ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
ax.set_yticklabels([0, '', 0.2, '', 0.4, '', 0.6])
ax.set_ylabel('spikelet rate (Hz)')
ax.set_xlim([-1, bar_x[-1]+2])
ax.spines['bottom'].set_visible(False)
plt.savefig(os.path.join(fig_folder, 'spikelet_rate_3states.png'), transparent=True)  


# stats for above plot
# do the paired boot stats against nost
base = 1  # index of nost in S
comp = [0, 2]  # indices of LIA, theta
num_b = 1000
p = np.full(len(comp), np.nan)
d = np.full(len(comp), np.nan)
for l in np.arange(len(comp)):
    dif = S[:, comp[l]] - S[:, base]
    # remove nans
    dif = dif[~np.isnan(dif)]
    d[l], p[l] = boot_pair_t(dif, num_b)
    print(dif.size)
print(d)
print(p)


# some descriptive numbers (omit big spikelet cells)
l = 0
np.nanmedian(S[:, l])
np.nanstd(S[:, l])
MADAM(S[:, l], np.nanmedian(S[:, l]))

#%% make figures - distributions of spikelet parameters

g0_cells = [12, 16]
g1_cells = [10, 50, 52, 56, 61, 67, 68, 88, 93, 100]
cond0 = df['cell_id'].isin(g0_cells)
cond1 = df['cell_id'].isin(g1_cells)

#measure = 'thresh_Vm'
for measure in ['amplitude', 'rise_time', 'decay_tau', 'thresh_Vm']:
    if measure == 'amplitude':
        bins = np.arange(0, 20, 2)
    if measure == 'fwhm':
        bins = np.arange(0, 1.6, 0.1)
    if measure == 'max_rise':
        bins = np.arange(5, 55, 5)
    if measure == 'rise_time':
        bins = np.arange(0, 1.2, 0.1)
    if measure == 'decay_tau':
        bins = np.arange(0, 1.1, 0.1)
    if measure == 'thresh_Vm':
        bins = np.arange(-80, -15, 5)
    fig, ax = plt.subplots(1, 1, figsize=[1.7, 1.7])
    ax.hist([df[measure][cond0], df[measure][cond1]], bins, color=c_spikelets, ec='none')
    if measure == 'amplitude':
        ax.set_xticks([0, 10, 20])
        ax.set_yticks([0, 100, 200, 300])
        ax.set_xlabel('amplitude (mV)')
    if measure == 'fwhm':
        ax.set_xticks([0, 0.5, 1.0, 1.5])
        ax.set_yticks([0, 50, 100, 150])
        ax.set_xlabel('FWHM (ms)')
    if measure == 'max_rise':
        ax.set_xticks([0, 20, 40])
        ax.set_yticks([0, 100, 200])
        ax.set_xlabel('max rise (V/s)')
    if measure == 'rise_time':
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_yticks([0, 100, 200])
        ax.set_xlabel('rise time (ms)')
    if measure == 'decay_tau':
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_yticks([0, 100, 200])
        ax.set_xlabel('decay time constant (ms)')
    if measure == 'thresh_Vm':
        ax.set_xticks([-80, -60, -40, -20])
        ax.set_yticks([0, 50, 100])
        ax.set_xlabel('threshold (mV)')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_folder, measure+'.png'), transparent=True)


# some descriptive numbers
measure = 'decay_tau'
len(df[measure][cond1])
np.nanmean(df[measure][cond1])
np.nanstd(df[measure][cond1])
MADAM(df[measure][cond1], np.nanmean(df[measure][cond1]))
len(df[measure][cond0])
np.nanmean(df[measure][cond0])
np.nanstd(df[measure][cond0])
MADAM(df[measure][cond0], np.nanmean(df[measure][cond0]))




