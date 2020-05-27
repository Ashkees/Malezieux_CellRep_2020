# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 17:10:52 2020

@author: ashley
"""


# Manuscript Malezieux, Kees, Mulle submitted to Cell Reports
# Figure S2 - theta coherence
# Description: changes in LFP/Vm theta power, coherence, and phase with resepct
# to the baseline Vm

# %% import modules

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex
import matplotlib as mpl
import pandas as pd


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

def boot_t_MADAM(t_g0, t_g1, num_b):
    real_d = MADAM(t_g1, np.nanmedian(t_g1)) - MADAM(t_g0, np.nanmedian(t_g0))
    faux_d = np.zeros(num_b)
    box = np.append(t_g0, t_g1)
    for b in np.arange(num_b):
        f_g0 = box[np.random.randint(0, box.size, size=t_g0.size)]
        f_g1 = box[np.random.randint(0, box.size, size=t_g1.size)]
        faux_d[b] = MADAM(f_g1, np.nanmedian(f_g1)) - MADAM(f_g0, np.nanmedian(f_g0))
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


## definition for removing subthreshold spike components from Vm_sub
## for this analysis, want a conservative estimate of average Vm, so will remove
## spikes and underlying depolarizations such as plateau potentials
## search_period is the time (in ms) over which to look for the end of the spike
## VERSION: put nan in place of the values that are taken out
#def remove_spikes_sub(Vm_sub_ts, Vm_sub, sp_times, search_period):
#    samp_rate = 1/(Vm_sub_ts[1]-Vm_sub_ts[0])
#    win = np.array(samp_rate*search_period/1000, dtype='int')
#    Vm_nosubsp = np.copy(Vm_sub)
#    sp_ind = np.searchsorted(Vm_sub_ts, sp_times)
#    for k in np.arange(sp_ind.size):
#        # only change Vm if it hasn't been already (i.e. for spikes in bursts)
#        if Vm_nosubsp[sp_ind[k]] == Vm_sub[sp_ind[k]]:
#            sp_end = np.array(Vm_nosubsp[sp_ind[k]:sp_ind[k]+win] >= Vm_sub[sp_ind[k]],
#                              float)
#            if np.all(sp_end == 1):
#                sp_end = sp_end.size
#            else:
#                sp_end = np.where(sp_end == 0)[0][0]
#            if sp_end > 0:
#                sp_end = sp_end+sp_ind[k]
#                # no need to interpolate, because start and end = Vm[sp_ind[i]]
#                #Vm_nosubsp[sp_ind[k]:sp_end] = Vm_sub[sp_ind[k]]
#                Vm_nosubsp[sp_ind[k]:sp_end] = np.nan
#    return Vm_nosubsp




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
fig_folder = r'C:\Users\akees\Documents\Ashley\Figures\2020-05_Paper_MIND1\FigS2'



#%% concatenate coh_df from all cells

# concatenate all the cell dataframes into one big dataframe
df = pd.concat([d['coh_df'] for d in data])

# remove entries that have NaNs for Ih or variance
df = df[~np.isnan(df['Ih'])]

# add a column to explicitly label hyp/dep/no events
dVm_type = np.full(df.shape[0], np.nan, dtype=object)
dVm_type[(df['dVm_p'] < 0.05) & (df['dVm'] < 0)] = 'hyp'
dVm_type[(df['dVm_p'] < 0.05) & (df['dVm'] > 0)] = 'dep'    
dVm_type[df['dVm_p'] > 0.05] = 'no'   
df['dVm_type'] = dVm_type   






# %% make figures - relationship between Vm and theta power (theta state only)

keep_cells = [isinstance(d['cell_id'], int) for d in data]

c_ntl_dk = [c_mgry, c_grn, c_bwn]

# bin over Vm for each cell
# don't separate by state or dVm type

Vm_bins = np.array([-80, -70, -60, -50, -40, -30])
#Vm_bins = np.array([-80, -50, -30])
#Vm_bins = np.arange(-80, -15, 5)
power_bin_Vm = np.full([len(data), Vm_bins.size-1], np.nan)
n = np.full([len(data), Vm_bins.size-1], np.nan)

for i in np.arange(len(data)):
    for v in np.arange(Vm_bins.size-1):
        cond1 = df['cell_id'] == data[i]['cell_id']
        cond2 = (df['Vm'] > Vm_bins[v]) & (df['Vm'] <= Vm_bins[v+1])
        cond3 = df['state'] == 'theta'
        values = df['power_Vm'][cond1 & cond2 & cond3]
        power_bin_Vm[i, v] = np.nanmedian(values)
        n[i, v] = values.size
power_bin_Vm = power_bin_Vm[keep_cells, :]
n = n[keep_cells, :]


# plot the cell variance values over binned Vm values
plot_Vm = Vm_bins[:-1] + 5
fig, ax = plt.subplots(1, 1, figsize=[3.4, 2.8])
for i in np.arange(power_bin_Vm.shape[0]):
    ax.plot(plot_Vm, power_bin_Vm[i, :], color=c_run_theta, alpha=0.5, zorder=2)
# add the cell median on top
ax.plot(plot_Vm, np.nanmedian(power_bin_Vm, axis=0), color=c_grn,
        linewidth=2, zorder=3)
ax.set_xlim([-80, -30])
ax.set_ylim([-0.2, 8])
ax.set_xticks([-80, -70, -60, -50, -40, -30])
ax.set_xlabel('Vm (mV)')
ax.set_yticks([0, 2, 4, 6, 8])
ax.set_ylabel('Vm theta power (V$^\mathrm{2}$)')
plt.tight_layout()
plt.savefig(os.path.join(fig_folder, 'power_vs_Vm.png'), transparent=True)

# stats
# compare each Vm bin to its neighbor (paired boot)
num_b = 1000
p = np.full(power_bin_Vm.shape[1]-1, np.nan)
d = np.full(power_bin_Vm.shape[1]-1, np.nan)
for v in np.arange(power_bin_Vm.shape[1]-1):
    dif = power_bin_Vm[:, v+1] - power_bin_Vm[:, v]
    # remove nans
    dif = dif[~np.isnan(dif)]
    d[v], p[v] = boot_pair_t(dif, num_b)
    print(dif.size)
print(d)
print(p)   

# descriptive numbers
np.sum(~np.isnan(power_bin_Vm), axis=0)
np.nanmedian(power_bin_Vm, axis=0)
np.nanstd(power_bin_Vm, axis=0)
for v in np.arange(power_bin_Vm.shape[1]):
    print(MADAM(power_bin_Vm[:, v], np.nanmedian(power_bin_Vm[:, v])))


# %% make figures - theta only, Vm relationship, separated over hyp/dep/no
# VERSION: raw power (not z-scored)


keep_cells = [isinstance(d['cell_id'], int) for d in data]
dVm_type = ['hyp', 'no', 'dep']
c_dVm_l = [c_lhyp, c_lgry, c_ldep]

# bin over Vm for each cell

Vm_bins = np.array([-80, -70, -60, -50, -40, -30])
#Vm_bins = np.array([-80, -50, -30])
#Vm_bins = np.arange(-80, -15, 5)
power_bin_Vm = np.full([len(data), Vm_bins.size-1, len(dVm_type)], np.nan)
n = np.full([len(data), Vm_bins.size-1, len(dVm_type)], np.nan)

for i in np.arange(len(data)):
    for m in np.arange(len(dVm_type)):
        for v in np.arange(Vm_bins.size-1):
            cond1 = df['cell_id'] == data[i]['cell_id']
            cond2 = (df['Vm'] > Vm_bins[v]) & (df['Vm'] <= Vm_bins[v+1])
            cond3 = df['state'] == 'theta'
            cond4 = df['dVm_type'] == dVm_type[m]
            values = df['power_Vm'][cond1 & cond2 & cond3 & cond4]
            power_bin_Vm[i, v, m] = np.nanmedian(values)
            n[i, v, m] = values.size
power_bin_Vm = power_bin_Vm[keep_cells, :, :]
n = n[keep_cells, :, :]

# plot the power values over binned Vm values
plot_Vm = Vm_bins[:-1] + 5
fig, ax = plt.subplots(1, 1, figsize=[1.7, 1.2])
for m in np.arange(len(dVm_type)):
#    for i in np.arange(power_bin_Vm.shape[0]):
#        ax.plot(plot_Vm, power_bin_Vm[i, :, m], color=c_dVm_l[m], alpha=0.5, zorder=2)
    # add the cell median on top
    ax.plot(plot_Vm, np.nanmedian(power_bin_Vm[:, :, m], axis=0), color=c_dVm[m],
            linewidth=2, zorder=3)
ax.set_xlim([-80, -30])
ax.set_ylim([0, 2])
#ax.set_xticks([-80, -70, -60, -50, -40, -30, -20])
ax.set_xticks([-70, -50, -30])
#ax.set_yticks([0, 2, 4, 6])
ax.set_yticks([0, 1, 2])
plt.tight_layout()
plt.savefig(os.path.join(fig_folder, 'power_vs_Vm_dVm_inset.png'), transparent=True)


# stats - put dep and no in same category
Vm_bins = np.array([-80, -70, -60, -50, -40, -30])
Vm_bins = np.array([-80, -50, -30])
#Vm_bins = np.arange(-80, -15, 5)
power_bin_Vm = np.full([len(data), Vm_bins.size-1, 2], np.nan)
n = np.full([len(data), Vm_bins.size-1, len(dVm_type)], np.nan)

for i in np.arange(len(data)):
    for m in [0, 1]:
        if m == 0:
            cond4 = df['dVm_type'] != 'hyp'
        if m == 1:
            cond4 = df['dVm_type'] == 'hyp'
        for v in np.arange(Vm_bins.size-1):
            cond1 = df['cell_id'] == data[i]['cell_id']
            cond2 = (df['Vm'] > Vm_bins[v]) & (df['Vm'] <= Vm_bins[v+1])
            cond3 = df['state'] == 'theta'
            values = df['power_Vm'][cond1 & cond2 & cond3 & cond4]
            power_bin_Vm[i, v, m] = np.nanmedian(values)
            n[i, v, m] = values.size
power_bin_Vm = power_bin_Vm[keep_cells, :, :]
n = n[keep_cells, :, :]

# stats - paired boot for each Vm bin
# hyp vs dep/no
num_b = 1000
p = np.full(power_bin_Vm.shape[1], np.nan)
d = np.full(power_bin_Vm.shape[1], np.nan)
for v in np.arange(power_bin_Vm.shape[1]):
    dif = power_bin_Vm[:, v, 1] - power_bin_Vm[:, v, 0]
    # remove nans
    dif = dif[~np.isnan(dif)]
    d[v], p[v] = boot_pair_t(dif, num_b)
    print(dif.size)
print(d)
print(p) 





# %% make figures - higher coherence during run theta than nonrun theta

keep_cells = [isinstance(d['cell_id'], int) for d in data]
theta_cell_p = np.array([d['theta_cell_p'] for d in data])[keep_cells]

c_rnr = [c_grn, c_run_theta]

#measure = 'coh_lfp'
measure = 'power_Vm'
S = np.full([len(data), 2], np.nan)
for i in np.arange(len(data)):
    cond1 = df['cell_id'] == data[i]['cell_id']
    cond2 = df['run_theta']
    values = df[measure][cond1 & cond2]
    S[i, 0] = np.nanmean(values)
    cond2 = (df['state'] == 'theta') & ~df['run_theta']
    values = df[measure][cond1 & cond2]
    S[i, 1] = np.nanmean(values)
S = S[keep_cells, :]

# plot the stackplots and boxplots over cells
fig, ax = plt.subplots(1, 1, figsize=[1.8, 1.7])
line_x = np.array([1.75, 3.25])
bar_x = np.array([1, 4])
# paired plot for mean
for i in np.arange(S.shape[0]):
    ax.plot(line_x, S[i, :], color=c_lgry, zorder=1)
    if theta_cell_p[i] < 0.05:
        ax.plot(line_x, S[i, :], color=rgb2hex(c_hyp), zorder=2)
    if theta_cell_p[i] > 0.95:
        ax.plot(line_x, S[i, :], color=rgb2hex(c_dep), zorder=2)
# boxplot for mean
for l in np.arange(S.shape[1]):
    # remove nans
    no_nan = S[:, l]
    no_nan = no_nan[~np.isnan(no_nan)]
    bp = ax.boxplot(no_nan, sym='', patch_artist=True,
                         whis=[5, 95], widths=0.75, positions=[bar_x[l]])     
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=c_rnr[l], linewidth=1.5)
    for patch in bp['boxes']:
        patch.set(facecolor=c_wht)
ax.set_xticks(bar_x)
ax.xaxis.set_tick_params(length=0)
ax.set_xticklabels(['run theta', 'rest theta'])
ax.set_xlim([0, bar_x[-1]+1])
ax.spines['bottom'].set_visible(False)
if measure == 'coh_lfp':
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1])
    ax.set_ylabel('Vm-LFP theta coherence')
if measure == 'coh_Vm':
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
if measure == 'power_Vm':
    ax.set_yticks([0, 2, 4])
    ax.set_ylabel('Vm theta power (V$^\mathrm{2}$)')
if measure == 'power_lfp':
    ax.set_yticks([0, 0.025, 0.05])
if measure == 'ph_lfp':
    ax.set_yticks([-180, -90, 0, 90, 180])
if measure == 'ph_Vm':
    ax.set_yticks([-180, -90, 0, 90, 180])
plt.tight_layout()
plt.savefig(os.path.join(fig_folder, measure+'_rnr.png'), transparent=True)


# stats
# compare each Vm bin to its neighbor (paired boot)
num_b = 1000
dif = S[:, 0] - S[:, 1]
# remove nans
dif = dif[~np.isnan(dif)]
print(dif.size)
boot_pair_t(dif, num_b)

# descriptive numbers
np.sum(~np.isnan(S), axis=0)
np.nanmedian(S, axis=0)
np.nanstd(S, axis=0)
for l in np.arange(S.shape[1]):
    print(MADAM(S[:, l], np.nanmedian(S[:, l])))






