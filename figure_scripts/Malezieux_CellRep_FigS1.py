# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 17:35:43 2018

@author: Ashley
"""


# Manuscript Malezieux, Kees, Mulle submitted to Cell Reports
# Figure S1 Pupil
# Description: changes in Vm with pupil dilations wrt state



# %% import modules

import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl


# %% definitions

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
    
    
# bootstrap: one-factor ANOVA-like:
# is between-group variance bigger than within-group?
def calculate_F4(g0, g1, g2, g3):
    box = np.append(np.append(g0, g1), np.append(g2, g3))
    GM = np.median(box)
    g0m = np.median(g0)
    g1m = np.median(g1)
    g2m = np.median(g2)
    g3m = np.median(g3)
    F = ((g0.size*np.abs(GM-g0m)+g1.size*np.abs(GM-g1m)+g2.size*np.abs(GM-g2m)+g3.size*np.abs(GM-g3m))/
          (np.sum(np.abs(g0-g0m))+np.sum(np.abs(g1-g1m))+np.sum(np.abs(g2-g2m))+np.sum(np.abs(g3-g3m))))
    return F

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



# %% load data
    
cell_files = ['cell_83.npy', 'cell_85.npy', 'cell_86.npy', 'cell_87.npy', 
              'cell_88.npy', 'cell_90.npy', 'cell_92.npy', 'cell_93.npy', 
              'cell_95.npy', 'cell_96.npy', 'cell_98.npy', 'cell_100.npy', ]
    
dataset_folder = (r'C:\Users\akees\Documents\Ashley\Papers\MIND 1\Cell Reports\Dryad upload\Dataset')

data = [{} for k in np.arange(len(cell_files))]
for i in np.arange(len(cell_files)):
    full_file = os.path.join(dataset_folder, cell_files[i])
    data[i] = np.load(full_file, allow_pickle=True).item()

# change pupil to be expressed in xminimum, instead of pixles
for i in np.arange(len(data)):
    data[i]['pupil'] = data[i]['pupil']/np.min(data[i]['pupil'])

     
states = [{'state':'theta', 'bef':-2.5, 'aft':0.5, 'samp_time':2, 't_win':[-4, 12]},
          {'state':'LIA', 'bef':-4, 'aft':-1, 'samp_time':2, 't_win':[-4, 12]},
          {'state':'run_theta', 'bef':-2.5, 'aft':0.5, 'samp_time':2, 't_win':[-4, 12]},
          {'state':'nonrun_theta', 'bef':-2.5, 'aft':0.5, 'samp_time':2, 't_win':[-4, 12]}]


# %% process data

# make a dictionary to hold values collapsed over all cells
events = [{} for k in np.arange(len(states))]
# find Vm0, dVm and significance for each run, excluding when Ih is changed
for l in np.arange(len(states)):
    all_c_p = np.empty(0)
    all_Ih = np.empty(0)
    all_Vm0 = np.empty(0)
    all_dVm = np.empty(0)
    all_dpup = np.empty(0)
    all_dVm_p = np.empty(0)
    for i in np.arange(len(data)):
        samp_freq = 1/(data[i]['Vm_ds_ts'][1] - data[i]['Vm_ds_ts'][0])
        samp_freq_pup = 1/(data[i]['pupil_ts'][1] - data[i]['pupil_ts'][0])
        num_ind = int(states[l]['samp_time']*samp_freq)
        num_ind_pup = int(states[l]['samp_time']*samp_freq_pup)
        # find index of dIh_times
        dIh_ind = data[i]['dIh_times']*samp_freq
        dIh_ind = dIh_ind.astype(int)
        c_p = np.zeros(data[i][states[l]['state']+'_start'].size)
        Ih = np.zeros(data[i][states[l]['state']+'_start'].size)
        Vm0 = np.zeros(data[i][states[l]['state']+'_start'].size)
        dVm = np.zeros(data[i][states[l]['state']+'_start'].size)
        dpup = np.zeros(data[i][states[l]['state']+'_start'].size)
        dVm_p = np.zeros(data[i][states[l]['state']+'_start'].size)
        for j in np.arange(data[i][states[l]['state']+'_start'].size):
            # find indices
            bef_ind = int(np.sum(data[i]['Vm_ds_ts'] <
                          (data[i][states[l]['state']+'_start'][j] + states[l]['bef'])))
            aft_ind = int(np.sum(data[i]['Vm_ds_ts'] <
                          (data[i][states[l]['state']+'_start'][j] + states[l]['aft'])))
            bef_ind_pup = int(np.sum(data[i]['pupil_ts'] <
                          (data[i][states[l]['state']+'_start'][j] + states[l]['bef'])))
            aft_ind_pup = int(np.sum(data[i]['pupil_ts'] <
                          (data[i][states[l]['state']+'_start'][j] + states[l]['aft'])))
            # put nan if times are straddling a time when dIh is changed
            dIh_true = np.where((dIh_ind > bef_ind) & (dIh_ind < aft_ind + num_ind))[0]
            if dIh_true.size > 0:
                Ih[j] = np.nan
                Vm0[j] = np.nan
                dVm[j] = np.nan
                dpup[j] = np.nan
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
#                if (np.nanmean(data[i]['pupil'][aft_ind_pup:aft_ind_pup+num_ind_pup]) - 
#                    np.nanmean(data[i]['pupil'][bef_ind_pup:bef_ind_pup+num_ind_pup])) > 0:
#                    dpup[j] = (np.nanmax(data[i]['pupil'][aft_ind_pup:aft_ind_pup+num_ind_pup]) - 
#                              np.nanmin(data[i]['pupil'][bef_ind_pup:bef_ind_pup+num_ind_pup]))
#                else:
#                    dpup[j] = (np.nanmin(data[i]['pupil'][aft_ind_pup:aft_ind_pup+num_ind_pup]) - 
#                              np.nanmax(data[i]['pupil'][bef_ind_pup:bef_ind_pup+num_ind_pup]))
#                start_ind = int(np.sum(data[i]['pupil_ts'] < (data[i][states[l]['state']+'_start'][j])))
#                stop_ind = int(np.sum(data[i]['pupil_ts'] < (data[i][states[l]['state']+'_stop'][j])))
#                if l == 1:
#                    dpup[j] = np.nanmean(data[i]['pupil'][start_ind:stop_ind])
#                else:
#                    dpup[j] = np.nanmean(data[i]['pupil'][start_ind:stop_ind])
                if l == 1:
                    dpup[j] = np.nanmin(data[i]['pupil'][aft_ind_pup:aft_ind_pup+num_ind_pup])
                else:
                    dpup[j] = np.nanmax(data[i]['pupil'][aft_ind_pup:aft_ind_pup+num_ind_pup])
        data[i][states[l]['state']+'_c_p'] = c_p
        data[i][states[l]['state']+'_Ih'] = Ih
        data[i][states[l]['state']+'_Vm0'] = Vm0
        data[i][states[l]['state']+'_dVm'] = dVm
        data[i][states[l]['state']+'_dpup'] = dpup
        data[i][states[l]['state']+'_dVm_p'] = dVm_p
        all_c_p = np.append(all_c_p, c_p)
        all_Ih = np.append(all_Ih, Ih)
        all_Vm0 = np.append(all_Vm0, Vm0)
        all_dVm = np.append(all_dVm, dVm)
        all_dpup = np.append(all_dpup, dpup)
        all_dVm_p = np.append(all_dVm_p, dVm_p)
    events[l]['c_p'] = all_c_p
    events[l]['Ih'] = all_Ih
    events[l]['Vm0'] = all_Vm0
    events[l]['dVm'] = all_dVm
    events[l]['dpup'] = all_dpup
    events[l]['dVm_p'] = all_dVm_p

# add pupil windows triggered by start of some brain states
for l in np.arange(len(states)):
    for i in np.arange(len(data)):
        t_pupil, t_pupil_ts = prepare_eta(data[i]['pupil'], data[i]['pupil_ts'],
                                 data[i][states[l]['state']+'_start'],
                                 states[l]['t_win'])
        t_Vm, t_Vm_ts = prepare_eta(data[i]['Vm_s_ds'], data[i]['Vm_ds_ts'],
                                 data[i][states[l]['state']+'_start'],
                                 states[l]['t_win'])
        data[i][states[l]['state']+'_pupil'] = t_pupil
        data[i][states[l]['state']+'_Vm'] = t_Vm
    states[l]['t_pupil_ts'] = t_pupil_ts
    states[l]['t_Vm_ts'] = t_Vm_ts
 

# add triggered windows to event dictionary
for l in np.arange(len(states)):
    event_nums = np.zeros(len(data))
    Vm = np.empty((t_Vm_ts.shape[0], 0))
    pupil = np.empty((t_pupil_ts.shape[0], 0))
    duration = np.empty(0)
    for i in np.arange(len(data)):
        if data[i][states[l]['state']+'_start'].size > 0:
            Vm = np.append(Vm, data[i][states[l]['state']+'_Vm'], axis=1)
            pupil = np.append(pupil, data[i][states[l]['state']+'_pupil'], axis=1)
            duration = np.append(duration, (data[i][states[l]['state']+'_stop'] -
                                     data[i][states[l]['state']+'_start']))
        event_nums[i] = Vm.shape[1]
    # remove nans
    no_nan = np.logical_and([~np.isnan(Vm).any(axis=0)],
                            [~np.isnan(pupil).any(axis=0)]).flatten()
    Vm = Vm[:, no_nan]
    pupil = pupil[:, no_nan]
    duration = duration[no_nan]
    events[l]['Vm'] = Vm
    events[l]['pupil'] = pupil
    events[l]['duration'] = duration
    events[l]['event_nums'] = event_nums
    events[l]['c_p'] = events[l]['c_p'][no_nan]
    events[l]['Ih'] = events[l]['Ih'][no_nan]
    events[l]['Vm0'] = events[l]['Vm0'][no_nan]
    events[l]['dVm'] = events[l]['dVm'][no_nan]
    events[l]['dpup'] = events[l]['dpup'][no_nan]
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
fig_folder = r'C:\Users\akees\Documents\Ashley\Figures\2020-05_Paper_MIND1\FigS1'


# %% panels for the actual figure


# plot the pupil color plot
l = 1
fig, ax = plt.subplots(1, 1, figsize=[1.7, 3.6])
# transpose the pupil values
norm_pupil = np.transpose(events[l]['pupil'])
# set order
order = np.flip(np.argsort(events[l]['duration']), axis=0)
p = ax.pcolormesh(states[l]['t_pupil_ts'], np.arange(order.size),
                  norm_pupil[order], cmap='BrBG', vmin=1, vmax=2)
ax.scatter(events[l]['duration'][order], np.arange(order.size)+0.5, color=c_blk, s=1)
ax.scatter(np.zeros(order.size), np.arange(order.size)+0.5, color=c_blk, s=1) 
ax.axis('tight')
ax.set_xticks([-4, 0, 4, 8, 12])
ax.set_xlabel('time relative to ' + states[l]['state'] + '\nonset (s)')
ax.set_ylim([0, order.size-1])
ax.set_yticks([order.size-1])
ax.set_yticklabels([order.size])
ax.set_ylabel('events', verticalalignment='center')
ax.yaxis.set_label_coords(-0.1, 0.5, transform=None)
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
divider = make_axes_locatable(ax)
cax = divider.append_axes("top", size="2%", pad=0.1)
cb = plt.colorbar(p, cax=cax, orientation="horizontal", ticks=[1, 2])
cb.set_label('pupil dilation\n(fold increase)', labelpad=-22)
cb.ax.xaxis.set_ticks_position('top')
cb.ax.xaxis.set_ticks([1, 2])
ax.set_xlim(states[l]['t_win'])
fig.tight_layout()
plt.savefig(os.path.join(fig_folder, states[l]['state'] + '_pupil_color.png'), transparent=True)




# example trace
# example cell 88, seconds 250-290
i = 4
fig, ax = plt.subplots(1, 1, figsize=[3.3, 1])
ax.plot(data[i]['pupil_ts'], data[i]['pupil'], color=c_blk)
for j in np.arange(data[i]['theta_start'].size):
    ax.axvspan(data[i]['theta_start'][j], data[i]['theta_stop'][j],
                  ymin=0.1, ymax=1, color=c_run_theta, alpha=1)
for j in np.arange(data[i]['LIA_start'].size):
    ax.axvspan(data[i]['LIA_start'][j], data[i]['LIA_stop'][j],
                  ymin=0.1, ymax=1, color=c_LIA, alpha=1)
ax.set_xlim([250, 290])
ax.set_ylim([0.9, 2])
ax.spines['left'].set_bounds(1, 2)
ax.spines['bottom'].set_bounds(280, 290)
ax.set_xlabel('10 s', horizontalalignment='center')
ax.xaxis.set_label_coords(0.875, -0.1, transform=None)
ax.xaxis.tick_bottom()
ax.set_yticks([1, 2])
ax.set_ylabel('fold\nincrease')
ax.set_xticks([])
ax.set_xticklabels([])
ax.set_title('pupil diameter', loc='left', fontsize=8)
fig.tight_layout()
plt.savefig(os.path.join(fig_folder, 'ex_pupil.png'), transparent=True)

