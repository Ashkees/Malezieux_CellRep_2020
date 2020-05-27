# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 15:14:12 2020

@author: ashley
"""

# Manuscript Malezieux, Kees, Mulle submitted to Cell Reports
# Figure 6 - variance
# Description: state-dependent relationship between Vm and variance

# %% import modules

import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from pingouin import ancova


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



#%% analysis across continuous time

# goal = dataframe with measures for each window

for i in np.arange(len(data)):
    # take every 20th datapoint from Vm_var, as these are independent
    # do the same for Vm_s_ds and Vm_ds_ts
    selected = np.arange(0, data[i]['Vm_ds_ts'].size, 20)
    Vm_var = data[i]['Vm_var'][selected]
    Vm_ds_ts = data[i]['Vm_ds_ts'][selected]
    Vm_s_ds = data[i]['Vm_s_ds'][selected]
    # save values as separate numpy arrays
    data[i]['var_Vm'] = Vm_s_ds
    data[i]['var_ts'] = Vm_ds_ts
    data[i]['var_var'] = Vm_var



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
    var_states = np.full(data[i]['var_ts'].size, 'nost', dtype='object')
    var_run_theta = np.zeros(data[i]['var_ts'].size, dtype='bool')
    Vm0 = np.full(data[i]['var_ts'].size, np.nan)
    dVm = np.full(data[i]['var_ts'].size, np.nan)
    dVm_p = np.full(data[i]['var_ts'].size, np.nan)
    # do LIA then theta, in case there are any windows that overlap
    # i.e. give priority to the theta values
    for l in [1, 0]:
        for j in np.arange(data[i][states[l]['id']+'_start'].size):
            ind0 = np.searchsorted(data[i]['var_ts'],
                                   data[i][states[l]['id']+'_start'][j])
            ind1 = np.searchsorted(data[i]['var_ts'],
                                   data[i][states[l]['id']+'_stop'][j])
            var_states[ind0:ind1] = states[l]['id']
            Vm0[ind0:ind1] = data[i][states[l]['id']+'_Vm0'][j]
            dVm[ind0:ind1] = data[i][states[l]['id']+'_dVm'][j]
            dVm_p[ind0:ind1] = data[i][states[l]['id']+'_dVm_p'][j]
            # cross check whether events match those in run_theta
            if np.any(np.isin(data[i]['run_theta_start'], data[i][states[l]['id']+'_start'][j])):
                var_run_theta[ind0:ind1] = True      
    # add the Ih
    inds = np.searchsorted(data[i]['Vm_Ih_ts'], data[i]['var_ts'])
    # if the last ind is the final ind, subtract 1
    if inds[-1] == data[i]['Vm_Ih_ts'].size:
        inds[-1] = inds[-1]-1
    Ih = data[i]['Vm_Ih'][inds]
    # put nans at the timestamp before and after the Ih change
    for h in np.arange(data[i]['dIh_times'].size):
        ind0 = np.searchsorted(data[i]['var_ts'], data[i]['dIh_times'][h])
        Ih[ind0] = np.nan
        Ih[ind0-1] = np.nan
    data[i]['var_states'] = var_states
    data[i]['var_run_theta'] = var_run_theta
    data[i]['var_Ih'] = Ih
    data[i]['var_Vm0'] = Vm0
    data[i]['var_dVm'] = dVm
    data[i]['var_dVm_p'] = dVm_p
 
    
# make the cell dataframes and concatenate into one big dataframe
for i in np.arange(len(data)):
    var_df = pd.DataFrame()
    var_df['Vm'] = data[i]['var_Vm']
    var_df['ts'] = data[i]['var_ts']
    var_df['var'] = data[i]['var_var']
    var_df['state'] = data[i]['var_states']
    var_df['run_theta'] = data[i]['var_run_theta']
    var_df['Ih'] = data[i]['var_Ih']
    var_df['Vm0'] = data[i]['var_Vm0']
    var_df['dVm'] = data[i]['var_dVm']
    var_df['dVm_p'] = data[i]['var_dVm_p']
    # add the cell id (include multiple recordings, but include under same label)
    # record the cell_p for theta and LIA from the first recording from the cell
    if isinstance(data[i]['cell_id'], str):
        ind = data[i]['cell_id'].find('_')
        cell_int = int(data[i]['cell_id'][:ind])
        var_df['cell_id'] = np.full(data[i]['var_ts'].size, cell_int)
        cell_ind = int(np.where(np.array([d['cell_id'] for d in data]) == str(cell_int))[0])
        var_df['theta_cell_p'] = np.full(data[i]['var_ts'].size, data[cell_ind]['theta_cell_p'])
        var_df['LIA_cell_p'] = np.full(data[i]['var_ts'].size, data[cell_ind]['LIA_cell_p'])
    else:
        cell_int = data[i]['cell_id']
        var_df['cell_id'] = np.full(data[i]['var_ts'].size, cell_int)
        var_df['theta_cell_p'] = np.full(data[i]['var_ts'].size, data[i]['theta_cell_p'])
        var_df['LIA_cell_p'] = np.full(data[i]['var_ts'].size, data[i]['LIA_cell_p']) 
    data[i]['var_df'] = var_df

# concatenate all the cell dataframes into one big dataframe
df = pd.concat([d['var_df'] for d in data])

# add a column to explicitly label hyp/dep/no events
dVm_type = np.full(df.shape[0], np.nan, dtype=object)
dVm_type[(df['dVm_p'] < 0.05) & (df['dVm'] < 0)] = 'hyp'
dVm_type[(df['dVm_p'] < 0.05) & (df['dVm'] > 0)] = 'dep'    
dVm_type[df['dVm_p'] > 0.05] = 'no'   
df['dVm_type'] = dVm_type   

# remove entries that have NaNs for Ih or variance
df = df[~np.isnan(df['Ih'])]
df = df[~np.isnan(df['var'])]


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

c_ntl = [c_lgry, c_run_theta, c_LIA]
c_ntl_dk = [c_mgry, c_grn, c_bwn]

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

keep_cells = [isinstance(d['cell_id'], int) for d in data]

# %% make figures - binned Vm, all theta or all LIA

#Vm_bins = np.array([-80, -70, -60, -50, -40, -30, -20])
Vm_bins = np.array([-80, -70, -60, -50, -40, -30])
var_bin_Vm = np.full([len(data), Vm_bins.size-1, len(ntl)], np.nan)
n = np.full([len(data), Vm_bins.size-1, len(ntl)], np.nan)

for i in np.arange(len(data)):
    for l in np.arange(len(ntl)):
        for v in np.arange(Vm_bins.size-1):
            cond1 = df['cell_id'] == data[i]['cell_id']
            cond2 = (df['Vm'] > Vm_bins[v]) & (df['Vm'] <= Vm_bins[v+1])
            cond3 = df['state'] == ntl[l]
            values = df['var'][cond1 & cond2 & cond3]
            # exclude variance values over 60
            values = values[values<60]
            var_bin_Vm[i, v, l] = np.nanmedian(values)
            n[i, v, l] = values.size
var_bin_Vm = var_bin_Vm[keep_cells, :, :]
n = n[keep_cells, :, :]



# plot the cell variance values over binned Vm values - theta only
plot_Vm = Vm_bins[:-1] + 5
fig, ax = plt.subplots(1, 1, figsize=[3.5, 2])
for l in [0, 1]:
    for i in np.arange(var_bin_Vm.shape[0]):
        ax.plot(plot_Vm, var_bin_Vm[i, :, l], color=c_ntl[l], alpha=0.25, zorder=1)
    # add the cell median on top
    ax.plot(plot_Vm, np.nanmedian(var_bin_Vm[:, :, l], axis=0), color=c_ntl_dk[l],
            linewidth=2, zorder=2)
ax.set_xlim([-80, -30])
ax.set_ylim([0, 17.5])
ax.set_xticks([-80, -70, -60, -50, -40, -30])
ax.set_xticklabels([-80, -70, -60, -50, -40, -30], fontsize=8)
ax.set_xlabel('Vm (mV)', fontsize=8)
ax.set_yticks([0, 5, 10, 15])
ax.set_yticklabels([0, 5, 10, 15], fontsize=8)
ax.set_ylabel('Vm variance (mV$^\mathrm{2}$)', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(fig_folder, 'all_cells_var_vs_Vm_theta.png'), transparent=True)        


# plot the cell variance values over binned Vm values - LIA only
plot_Vm = Vm_bins[:-1] + 5
fig, ax = plt.subplots(1, 1, figsize=[3, 1.8])
for l in [0, 2]:
    for i in np.arange(var_bin_Vm.shape[0]):
        ax.plot(plot_Vm, var_bin_Vm[i, :, l], color=c_ntl[l], alpha=0.25, zorder=1)
    # add the cell median on top
    ax.plot(plot_Vm, np.nanmedian(var_bin_Vm[:, :, l], axis=0), color=c_ntl_dk[l],
            linewidth=2, zorder=2)
ax.set_xlim([-80, -30])
ax.set_ylim([0, 17.5])
ax.set_xticks([-80, -70, -60, -50, -40, -30])
ax.set_xticklabels([-80, -70, -60, -50, -40, -30], fontsize=8)
ax.set_xlabel('Vm (mV)', fontsize=8)
ax.set_yticks([0, 5, 10, 15])
ax.set_yticklabels([0, 5, 10, 15], fontsize=8)
ax.set_ylabel('Vm variance (mV$^\mathrm{2}$)', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(fig_folder, 'all_cells_var_vs_Vm_LIA.png'), transparent=True)   





# %% try stats - ancova

# try the ancova using pingouin
ancova_results = ancova(data=df, dv='var',
                        covar=['Vm'], between='state')
# for just nost and theta
ancova_results = ancova(data=df[df['state'] != 'LIA'], dv='var',
                        covar=['Vm'], between='state')
# for just nost and theta, theta hyp only
ancova_results = ancova(data=df[(df['state'] != 'LIA') & (df['dVm_type'] == 'hyp')],
                        dv='var', covar=['Vm'], between='state')

# %% stats - paired stats by cell for slope and intercept

# I think this is the best stats for this
# linear regression of variance vs Vm for each cell, then compare slopes and
# intercepts with paired bootstrap for nost vs theta and nost vs LIA
S_slope = np.full([len(data), len(ntl)], np.nan)
S_intcpt = np.full([len(data), len(ntl)], np.nan)
n = np.full([len(data), len(ntl)], np.nan)

# do the linear regression for each cell and each state
for i in np.arange(len(data)):
    for l in np.arange(len(ntl)):
        x = df['Vm'][(df['cell_id'] == data[i]['cell_id']) & (df['state'] == ntl[l])]
        y = df['var'][(df['cell_id'] == data[i]['cell_id']) & (df['state'] == ntl[l])]
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
    # remove nans
    dif = dif[np.logical_and((~np.isnan(S_slope[:, l+1])), (~np.isnan(S_slope[:, 0])))]
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
    # remove nans
    dif = dif[np.logical_and((~np.isnan(S_slope[:, l+1])), (~np.isnan(S_slope[:, 0])))]
    d[l], p[l] = boot_pair_t(dif, num_b)
    print(dif.size)
print(d)
print(p)


# some descriptive numbers
l = 2
np.nanmedian(S_slope[:, l])
np.nanstd(S_slope[:, l])
MADAM(S_slope[:, l], np.nanmedian(S_slope[:, l]))
np.nanmedian(S_intcpt[:, l])
np.nanstd(S_intcpt[:, l])
MADAM(S_intcpt[:, l], np.nanmedian(S_intcpt[:, l]))




