# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 13:14:01 2019

@author: Ashley
"""


# Manuscript Malezieux, Kees, Mulle submitted to Cell Reports
# Figure S7
# Description: mechanisms of LIA depolarization: input resistance


# %% load modules
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import random
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


#prepare the Rin measure-triggered average to plot the example traces
#Rin measure-triggered average
def prepare_eta(signal, ts, event_times, win):
    win_npts = [ts[ts < ts[0] + np.abs(win[0])].size,
                ts[ts < ts[0] + np.abs(win[1])].size]
    et_ts = ts[0:np.sum(win_npts)] - ts[0] + win[0]
    # remove any events that are too close to the beginning or end of recording
    if event_times[0]+win[0] < ts[0]:
        event_times = event_times[1:-1]
    if event_times[-1]+win[1] > ts[-1]:
        event_times = event_times[0:-2]
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
    

dataset_folder = (r'C:\Users\akees\Documents\Ashley\Papers\MIND 1\Cell Reports\Dryad upload\Dataset_Rin')

cell_files = os.listdir(dataset_folder)
data = [{} for k in np.arange(len(cell_files))]
for i in np.arange(len(cell_files)):
    full_file = os.path.join(dataset_folder, cell_files[i])
    data[i] = np.load(full_file, allow_pickle=True).item()



state = ['theta', 'LIA']


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
fig_folder = r'C:\Users\akees\Documents\Ashley\Figures\2020-05_Paper_MIND1\FigS7'


# %% make figures

# for each cell, run a 1-way anova over the Rin values for theta, LIA and nost
state_Rin_anova_p = np.full(len(data), np.nan)
state_Rin_t_p = np.full([len(data), len(state)], np.nan)
state_Rin_t_d = np.full([len(data), len(state)], np.nan)
for i in np.arange(len(data)):
    groups_list = [data[i]['Rin'][data[i]['nost_Rin']],
                   data[i]['Rin'][data[i]['theta_Rin']],
                   data[i]['Rin'][data[i]['LIA_Rin']]]
    real_F, p_boot_anova = boot_anova(groups_list, 1000)
    state_Rin_anova_p[i] = p_boot_anova
    for l in np.arange(len(state)):
        real_d, p = boot_t(data[i]['Rin'][data[i]['nost_Rin']],
                           data[i]['Rin'][data[i][state[l]+'_Rin']], 1000)
        state_Rin_t_p[i, l] = p
        state_Rin_t_d[i, l] = real_d
# 6 out of 13 cells have significant differences

# cell id of the cells that change Rin significantly
l = 1
np.array([d['cell_id'] for d in data])[state_Rin_t_p[:, l] < 0.05]


## no state Rin vs theta or LIA Rin
#for l in np.arange(len(state)):
#    fig, ax = plt.subplots(1, figsize=[1.75, 1.75])
#    for i in np.arange(len(data)):
#        x = np.nanmean(data[i]['Rin'][data[i]['nost_Rin']])
#        y = np.nanmean(data[i]['Rin'][data[i][state[l]+'_Rin']])
#        ax.plot([0,400], [0,400], color=c_blk, zorder=1)
#        ax.scatter(x, y, facecolors='none', edgecolors=c_state[l], linewidth=1.5,
#                   zorder=2)
#        if state_Rin_t_p[i, l] < 0.05:
#            ax.scatter(x, y, facecolors=c_state[l], edgecolors=c_state[l],
#                       linewidth=1.5, zorder=2)
#    ax.set_xlim([0, 400])
#    ax.set_xticks([0, 200, 400])
#    ax.set_xticklabels([])
#    ax.set_ylim([0, 405])
#    ax.spines['left'].set_bounds(0, 400)
#    ax.set_yticks([0, 200, 400])
#    ax.set_yticklabels([])
#    fig.tight_layout()
#    plt.savefig(os.path.join(fig_folder, 'Rin_scatter_'+state[l]+'.png'),
#                             transparent=True)



# make a stack and bar plot of Rin (change from nost) - LIA only
l = 1
s_cell = 40
#fig, ax = plt.subplots(1, figsize=[0.8, 1.5])
# create a figure with axes of defined size
fig = plt.figure(figsize=[2, 2])
# The first items are for padding and the second items are for the axes.
# sizes are in inch.
h = [Size.Fixed(0.5), Size.Fixed(0.6)]
v = [Size.Fixed(0.5), Size.Fixed(1.1)]
divider = Divider(fig, (0.0, 0.0, 1., 1.), h, v, aspect=False)
ax = Axes(fig, divider.get_position())
ax.set_axes_locator(divider.new_locator(nx=1, ny=1))
fig.add_axes(ax)
for i in np.arange(len(data)):
    cell_Rin_nost = np.nanmean(data[i]['Rin'][data[i]['nost_Rin']])
    cell_Rin_state = np.nanmean(data[i]['Rin'][data[i][state[l]+'_Rin']])
    cell_Rin_state = (cell_Rin_state-cell_Rin_nost)/cell_Rin_nost
    if state_Rin_t_p[i, l] < 0.05:
        ax.scatter(random.uniform(0, 1), cell_Rin_state, facecolors=c_state[l],
                    edgecolors=c_state[l], zorder=2, s=s_cell)
    else:
        ax.scatter(random.uniform(0, 1), cell_Rin_state, facecolors='none',
                   edgecolors=c_state[l], zorder=2, s=s_cell)
    ax.plot([-1, 2], [0, 0], color=c_blk, linewidth=0.8, linestyle='--', zorder=1)
ax.set_xlim([-0.2, 1.3])
ax.set_xticks([])
ax.axes.spines['bottom'].set_visible(False)
ax.set_ylim([-0.08, 0.33])
ax.set_yticks([0, 0.1, 0.2, 0.3])
ax.set_yticklabels([0, 10, 20, 30])
ax.set_ylabel(r'$\Delta$ input resistance (%)')
plt.savefig(os.path.join(fig_folder, 'Rin_stack_'+state[l]+'.png'),
                         transparent=True)


# make a stack and bar plot of theta and Rin (change from nost)
for l in np.arange(len(state)):
    fig, ax = plt.subplots(1, figsize=[0.5, 1])
    for i in np.arange(len(data)):
        cell_Rin_nost = np.nanmean(data[i]['Rin'][data[i]['nost_Rin']])
        cell_Rin_state = np.nanmean(data[i]['Rin'][data[i][state[l]+'_Rin']])
        cell_Rin_state = (cell_Rin_state-cell_Rin_nost)/cell_Rin_nost
        if state_Rin_t_p[i, l] < 0.05:
            ax.scatter(random.uniform(0, 1), cell_Rin_state, facecolors=c_state[l],
                        edgecolors=c_state[l], zorder=2)
        else:
            ax.scatter(random.uniform(0, 1), cell_Rin_state, facecolors='none',
                       edgecolors=c_state[l], zorder=2)
        ax.plot([-1, 2], [0, 0], color=c_blk)
    ax.set_xlim([-0.5, 1.5])
    ax.set_xticks([])
    ax.axes.spines['bottom'].set_visible(False)
    ax.set_ylim([-0.4, 0.4])
    ax.set_yticks([-0.4, -0.2, 0, 0.2, 0.4])
    ax.set_yticklabels([])
    plt.savefig(os.path.join(fig_folder, 'Rin_stack_'+state[l]+'.png'),
                             transparent=True)

    
# get descriptive stats from above plot
# prepare numbers
cell_Rin = np.full([len(data), len(state)+1], np.nan)
for l in np.arange(len(state)):
    for i in np.arange(len(data)):
        nost_ind = np.where(data[i]['nost_Rin'])[1]
        cell_Rin[i, 0] = np.nanmean(data[i]['Rin'][nost_ind])
        st_ind = np.where(data[i][state[l]+'_Rin'])[1]
        cell_Rin[i, l+1] = np.nanmean(data[i]['Rin'][st_ind])
# paired stats on cell Rin values
cell_Rin_change = np.full([len(data), len(state)], np.nan)
num_b = 1000
real_d = np.full(len(state), np.nan)
p_pair_boot = np.full(len(state), np.nan)
for l in np.arange(len(state)):
    dif = cell_Rin[:, l+1] - cell_Rin[:, 0]
    real_d[l], p_pair_boot[l] = boot_pair_t(dif, num_b)
    cell_Rin_change[:, l] = dif/cell_Rin[:, 0]
# descriptive stats on % change values
np.median(cell_Rin_change, axis=0)
np.std(cell_Rin_change, axis=0)
# theta  
MADAM(cell_Rin_change[:, 0], np.median(cell_Rin_change[:, 0]))
# LIA
MADAM(cell_Rin_change[:, 1], np.median(cell_Rin_change[:, 1]))


