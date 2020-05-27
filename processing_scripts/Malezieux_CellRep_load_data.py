# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 10:23:49 2018

@author: Ashley
"""


# Manuscript Malezieux, Kees, Mulle submitted to eLife
# initial processing of data; brain state detection


# %% import modules

import os
import numpy as np
import pandas as pd
from neo import io
from scipy import signal
from scipy import stats


# %% load excel sheet

# load Data_to_Analyze.xlsx to to be able to load data
sheets = ['MIND-1 v3']
cells = np.empty(0)
mouse_ids = np.empty(0)
data_folder = np.empty(0)
ipdep_file = np.empty(0)
stepRa_file = np.empty(0)
ephy_file = np.empty(0)
ephy_file = np.empty(0)
wh_file = np.empty(0)
eye_track = np.empty(0)
sweep_lenght = np.empty(0)
good_seconds_start = np.empty(0)
good_seconds_stop = np.empty(0)
for i in np.arange(len(sheets)):
    data_list = pd.read_excel(r"C:\Users\akees\Documents\Ashley\Analysis\MIND"
                              r"\Data_to_Analyze_exHD.xlsx",
                              sheet_name=sheets[i])
    cells = np.append(cells, data_list.loc[:, 'cell #'].values)
    mouse_ids = np.append(mouse_ids, data_list.loc[:, 'mouse ID'].values)
    data_folder = np.append(data_folder, data_list.loc[:, 'Raw Data Folder'].values)
    ipdep_file = np.append(ipdep_file, data_list.loc[:, 'ip_dep step'].values)
    stepRa_file = np.append(stepRa_file, data_list.loc[:, 'step Ra'].values)
    ephy_file = np.append(ephy_file, data_list.loc[:, 'ephy file'].values)
    wh_file = np.append(wh_file, data_list.loc[:, 'wheel file'].values)
    good_seconds_start = np.append(good_seconds_start,
                                   data_list.loc[:, 'good data start (seconds)'].values)
    good_seconds_stop = np.append(good_seconds_stop,
                                  data_list.loc[:, 'good data end (seconds)'].values)
    sweep_lenght = np.append(sweep_lenght,
                             data_list.loc[:, 'Length of sweeps (s)'].values)
    eye_track = np.append(eye_track,
                          data_list.loc[:, 'Eye tracking file'].values)





# %% definitions

# definition for downsampling
def ds(ts, signal, ds_factor):
    signal_ds = np.mean(np.resize(signal,
                        (int(np.floor(signal.size/ds_factor)), ds_factor)), 1)
    ds_ts = ts[np.arange(int(np.round(ds_factor/2)), ts.size, ds_factor)]
    # trim off last time stamp if necessary
    ds_ts = ds_ts[0:signal_ds.size]
    return ds_ts, signal_ds

# load data - LFP 200 Hz, running 32 Hz, pupil 10 Hz, raw Vm 20 kHz
# can handle both episodic and gap free
# VERSION: fixed an invisible bug that appears when you don't keep the first
# sweeps in an episodic protocol
# VERSION: fixed some bugs in the loading and processing of pupil diameter
# VERSION: allows good_seconds_start/stop to be within sweeps
def load_data_MIND(cell_ind):

    i = cell_ind

    # load some Axon data from ABF files
    file_name = os.path.join(data_folder[i], ephy_file[i])
    # r is the name bound to the object created by io.AxonIO
    r = io.AxonIO(filename=file_name)
    # bl is the object that actually has the data, created by read_block
    bl = r.read_block()

    # get list of channel names
    channel_list = []
    for asig in bl.segments[0].analogsignals:
        channel_list.append(asig.name)
    
    if np.isnan(sweep_lenght[i]):
        full_ts = np.copy(bl.segments[0].analogsignals[0].times)
        lfp_raw = np.copy(bl.segments[0].analogsignals[1].data)
        lfp_raw = lfp_raw[(full_ts >= good_seconds_start[i]) &
                          (full_ts < good_seconds_stop[i])]
        lfp_raw = np.squeeze(lfp_raw)
        Vm = np.copy(bl.segments[0].analogsignals[0].data)
        Vm = Vm[(full_ts >= good_seconds_start[i]) &
                (full_ts < good_seconds_stop[i])]
        Vm = np.squeeze(Vm)
        Vm_ts = full_ts[(full_ts >= good_seconds_start[i]) &
                        (full_ts < good_seconds_stop[i])]
    else: 
        sweep_end = len(bl.segments)
        sweep_pts = len(bl.segments[0].analogsignals[0].times)
        full_ts = np.zeros(sweep_pts*sweep_end) 
        for j in np.arange(sweep_end):
            start_ind = j*sweep_pts
            a = np.squeeze(bl.segments[j].analogsignals[0].times)
            full_ts[start_ind:start_ind+sweep_pts] = a
        lfp_raw = np.zeros(sweep_pts*sweep_end)
        for k in np.arange(sweep_end):
            start_ind = k*sweep_pts
            a = np.squeeze(bl.segments[k].analogsignals[1].data)
            lfp_raw[start_ind:start_ind+sweep_pts] = a
        Vm = np.zeros(sweep_pts*sweep_end)
        for l in np.arange(sweep_end):
            start_ind = l*sweep_pts
            a = np.squeeze(bl.segments[l].analogsignals[0].data)
            Vm[start_ind:start_ind+sweep_pts] = a
        # remove the times that we don't want
        lfp_raw = lfp_raw[(full_ts >= good_seconds_start[i]) &
                          (full_ts < good_seconds_stop[i])]
        Vm = Vm[(full_ts >= good_seconds_start[i]) &
                (full_ts < good_seconds_stop[i])]
        Vm_ts = full_ts[(full_ts >= good_seconds_start[i]) &
                        (full_ts < good_seconds_stop[i])]
            
        
    # downsample lfp to 2000 Hz (factor of 10)
    ds_factor = 10
    lfp_ds_ts_10, lfp_ds_10 = ds(Vm_ts, lfp_raw, ds_factor)

    samp_freq = 1/(lfp_ds_ts_10[1] - lfp_ds_ts_10[0])
    nyq = samp_freq/2

    # filter the lfp between 0.2 Hz and 100 Hz
    # this algorithm seems to cause no time shift
    # high pass filter
    b, a = signal.butter(4, 0.2/nyq, "high", analog=False)
    lfp_highpass = signal.filtfilt(b, a, lfp_ds_10)
    # low pass filter
    wp = 80  # Hz, passband
    ws = 120  # Hz, stopband
    N, Wn = signal.buttord(wp/nyq, ws/nyq, 3, 40)
    b, a = signal.butter(N, Wn, "low", analog=False)
    lfp_f = signal.filtfilt(b, a, lfp_highpass)
    
    # downsample lfp a second time to 200 Hz (factor of 10)
    ds_factor = 10
    lfp_ts, lfp = ds(lfp_ds_ts_10, lfp_f, ds_factor)

    # if the file has synchronization info for the wheel, use it.
    # If not, keep the original timestamps.
    # keep only the selection of wheel data according to the good seconds
    if 'IN7' in channel_list:
        ind = channel_list.index('IN7')
        if np.isnan(sweep_lenght[i]):
            TTL = np.squeeze(np.copy(bl.segments[0].analogsignals[ind].data))
        else:
            TTL = np.zeros(sweep_pts*sweep_end)
            for l in np.arange(sweep_end):
                start_ind = l*sweep_pts
                a = np.squeeze(bl.segments[l].analogsignals[ind].data)
                TTL[start_ind:start_ind+sweep_pts] = a
        # find the axon times where the 32 Hz goes from high V to low
        wh_ts = full_ts[np.ediff1d(1*(TTL < 1), to_begin=0) > 0]
        # something is weird - I would have thought it should be < 0

        # load the corresponding wheel file (ignore imtrk timestamps)
        file_name = os.path.join(data_folder[i], wh_file[i])
        imtrk = pd.read_excel(file_name)
        wh_speed = imtrk.values[:, 1]  # as calculated by imetronic
        # if wheel file is longer than ephy, trim off the end
        wh_speed = wh_speed[0:wh_ts.size]
        # save only the good seconds according to the excel file
        wh_speed = wh_speed[(wh_ts >= good_seconds_start[i]) &
                            (wh_ts < good_seconds_stop[i])]
        wh_ts = wh_ts[(wh_ts >= good_seconds_start[i]) &
                      (wh_ts < good_seconds_stop[i])]
    else:
        file_name = os.path.join(data_folder[i], wh_file[i])
        imtrk = pd.read_excel(file_name)
        wh_ts = imtrk.values[:, 0]/1000  # in seconds, sampled at 32Hz
        wh_speed = imtrk.values[:, 1]  # as calculated by imetronic
        wh_speed = wh_speed[(wh_ts >= good_seconds_start[i]) &
                            (wh_ts < good_seconds_stop[i])]
        wh_ts = wh_ts[(wh_ts >= good_seconds_start[i]) &
                      (wh_ts < good_seconds_stop[i])]
        
    # load the extracted pupil diameters, use synchronization timestamps
    if 'IN5' in channel_list:
        ind = channel_list.index('IN5')
        if isinstance(eye_track[i], str):
            TTL = np.zeros(sweep_pts*(sweep_end))
            for j in np.arange(sweep_end):
                start_ind = j*sweep_pts
                a = np.squeeze(bl.segments[j].analogsignals[ind].data)
                TTL[start_ind:start_ind+sweep_pts] = a
            pupil_ts = full_ts[np.ediff1d(1*(TTL < 1), to_begin=0) > 0]
            file_name = os.path.join(data_folder[i], eye_track[i])
            pupil_excel = pd.read_excel(file_name)
            radii = pupil_excel.iloc[:, 1].values
            radii = radii[0:pupil_ts.size]
            radii = radii[(pupil_ts >= good_seconds_start[i]) &
                          (pupil_ts < good_seconds_stop[i])]
            pupil_ts = pupil_ts[(pupil_ts >= good_seconds_start[i]) &
                                (pupil_ts < good_seconds_stop[i])]
            radii_nozero = np.copy(radii)
            for j in np.arange(radii.size):
                if radii[j] == 0:
                    radii_nozero[j] = radii_nozero[j-1]
            # low pass filter
            c, d = signal.butter(4, 0.1, "low", analog=False)
            # 4 poles, 0.5 Hz normalized by nyquist of 5 is 0.1
            pupil = signal.filtfilt(c, d, radii_nozero)
        else:
            pupil = np.empty(0)
            pupil_ts = np.empty(0)
    else:
        pupil = np.empty(0)
        pupil_ts = np.empty(0)
        
    # if the file has 'Chan2Hold', load it, if not, create a nan vector
    ds_factor = 10000
    if 'Chan2Hold' in channel_list:
        ind = channel_list.index('Chan2Hold')
        if np.isnan(sweep_lenght[i]):
           Vm_Ih = np.squeeze(np.copy(bl.segments[0].analogsignals[ind].data))
           Vm_Ih = Vm_Ih[(full_ts >= good_seconds_start[i]) &
                         (full_ts < good_seconds_stop[i])] 
        else:
            Vm_Ih = np.zeros(sweep_pts*sweep_end)
            for f in np.arange(sweep_end):
                start_ind = f*sweep_pts
                a = np.squeeze(bl.segments[f].analogsignals[ind].data)
                Vm_Ih[start_ind:start_ind+sweep_pts] = a
            # keep only the good seconds
            Vm_Ih = Vm_Ih[(full_ts >= good_seconds_start[i]) &
                         (full_ts < good_seconds_stop[i])] 
        # downsample Vm_Ih to 2 Hz (factor of 10000)
        Vm_Ih = np.mean(np.resize(Vm_Ih,
                        (int(np.floor(Vm_Ih.size/ds_factor)), ds_factor)), 1)
        Vm_Ih_ts = Vm_ts[np.arange(0, Vm_ts.size, ds_factor)]
        # trim off last time stamp if necessary
        Vm_Ih_ts = Vm_Ih_ts[0:Vm_Ih.size]
    else:
        Vm_Ih = np.empty(int(Vm_ts.size/ds_factor))
        Vm_Ih[:] = np.nan
        Vm_Ih_ts = Vm_ts[np.arange(0, Vm_ts.size, ds_factor)]
        # trim off last time stamp if necessary
        Vm_Ih_ts = Vm_Ih_ts[0:Vm_Ih.size]

    return Vm_ts, Vm, lfp_ts, lfp, wh_ts, wh_speed, pupil_ts, pupil, Vm_Ih_ts, Vm_Ih



# definition for finding run times
def find_run_times(wh_ts, wh_speed):
    nonzero_speed = np.array(wh_speed > 0, float)
    run_start = wh_ts[np.ediff1d(nonzero_speed, to_begin=0) == 1]
    run_stop = wh_ts[np.ediff1d(nonzero_speed, to_begin=0) == -1]
    if np.logical_or(run_start.size==0, run_stop.size==0):
        run_start = np.empty(0)
        run_stop = np.empty(0)
    else:
        # remove runs that occur either at very start or very end of recording
        if run_start[0]-run_stop[0] > 0:
            run_stop = run_stop[1:]
        if run_start.shape != run_stop.shape:
            run_start = run_start[0:-1]
    return run_start, run_stop


# definition of preparation of z-scored Sxx
def prepare_Sxx(lfp_ts, lfp, win, overlap, frange):
    pad_factor = 2**2
    samp_rate = 1/(lfp_ts[1]-lfp_ts[0])
    nperseg = int(2**(np.round(np.log2(win*samp_rate))))
    f, spec_ts, Sxx = signal.spectrogram(lfp, fs=samp_rate,
                                         window='hamming', nperseg=nperseg,
                                         axis=0, noverlap=nperseg*overlap,
                                         nfft=pad_factor*nperseg,
                                         scaling='density')
    Sxx = np.squeeze(Sxx)
    spec_ts = spec_ts + lfp_ts[0]
    theta = [6, 9]
    delta = [0.5, 3.5]
    theta_power = np.mean(Sxx[(f > theta[0]) & (f < theta[1]), :], 0)
    delta_power = np.mean(Sxx[(f > delta[0]) & (f < delta[1]), :], 0)
    theta_delta = stats.zscore(theta_power/delta_power)
    z_Sxx = stats.zscore(Sxx[(f > frange[0]) & (f < frange[1]), :], axis=1)
    f = f[(f > frange[0]) & (f < frange[1])]
    return z_Sxx, f, spec_ts, theta_delta


# definition for how to extract spectral features from z_Sxx
def lfp_features(spec_ts, f, Sxx, break_ind):
    # average power over frequency bands
    delta = [0.5, 3.5]
    delta_power = np.mean(Sxx[(f > delta[0]) & (f < delta[1]), :], 0)
    delta_power = signal.detrend(delta_power, bp=break_ind)
    theta = [6, 9]
    theta_power = np.mean(Sxx[(f > theta[0]) & (f < theta[1]), :], 0)
    theta_power = signal.detrend(theta_power, bp=break_ind)
    beta = [10, 20]
    beta_power = np.mean(Sxx[(f > beta[0]) & (f < beta[1]), :], 0)
    beta_power = signal.detrend(beta_power, bp=break_ind)
    gamma = [30, 50]
    gamma_power = np.mean(Sxx[(f > gamma[0]) & (f < gamma[1]), :], 0)
    gamma_power = signal.detrend(gamma_power, bp=break_ind)
    total = [0.3, 80]
    total_power = np.mean(Sxx[(f > total[0]) & (f < total[1]), :], 0)
    total_power = signal.detrend(total_power, bp=break_ind)
    return delta_power, theta_power, beta_power, gamma_power, total_power

  
# find timestamps where running is nonzero
def find_running_bool(ts, run_start, run_stop):
    run_bool = np.zeros(ts.size, dtype='bool')
    # find timestamps closest to running start and stop times
    for i in np.arange(run_start.size):
        ind_start = np.argmin(np.abs(ts-run_start[i]))
        ind_stop = np.argmin(np.abs(ts-run_stop[i]))
        run_bool[ind_start:ind_stop] = True
    return run_bool


# definition for finding spike indices
# VERSION: special case for 1 spike
# VERSION: round samp_rate to avoid error in ediff1d
def find_sp_ind(Vm_ts, Vm, thresh, refrac):
    "Returns the indices of spikes.  Refrac is in ms"
    Vm_diff = np.reshape(Vm, (1, Vm.size))
    dVdt = np.diff(Vm_diff)
    dVdt = np.reshape(dVdt, (dVdt.size))
    # show the threshold and where the spike would be detected
    dVdt_thresh = np.array(dVdt > thresh, float)
    if sum(dVdt_thresh) == 0:
        # there are no spikes
        sp_ind = np.empty(shape=0)
    elif sum(dVdt_thresh) == 1:
        sp_ind = np.where(np.diff(dVdt_thresh) == 1)[0]
    else:
        # keep just the first index per spike
        sp_ind = np.squeeze(np.where(np.diff(dVdt_thresh) == 1))
        # remove any duplicates of spikes that occur within refractory period
        samp_rate = np.round(1/(Vm_ts[1]-Vm_ts[0]))
        sp_ind = sp_ind[np.ediff1d(sp_ind,
                        to_begin=samp_rate*refrac/1000+1) >
                        samp_rate*refrac/1000]
    return sp_ind


# definition for removing spikes
# for this analysis, want a conservative estimate of average Vm, so will remove
# spikes and underlying depolarizations such as plateau potentials
# search_period is the time (in ms) over which to look for the end of the spike
def remove_spikes(Vm_ts, Vm, sp_ind, search_period):
    samp_rate = 1/(Vm_ts[1]-Vm_ts[0])
    win = np.array(samp_rate*search_period/1000, dtype='int')
    Vm_nosp = np.copy(Vm)
    for k in np.arange(sp_ind.size):
        # only change Vm if it hasn't been already (i.e. for spikes in bursts)
        if Vm_nosp[sp_ind[k]] == Vm[sp_ind[k]]:
            sp_end = np.array(Vm_nosp[sp_ind[k]:sp_ind[k]+win] > Vm[sp_ind[k]],
                              float)
            sp_end, = np.where(np.diff(sp_end) == -1)
            if sp_end.size > 0:
                sp_end = sp_end[0]+sp_ind[k]
                # no need to interpolate, because start and end = Vm[sp_ind[i]]
                Vm_nosp[sp_ind[k]:sp_end] = Vm[sp_ind[k]]
    return Vm_nosp 


# definition for finding times where the Ih is changed
def find_dIh_times(Vm_Ih_ts, Vm_Ih):
    dIh = np.abs(np.ediff1d(Vm_Ih, to_begin=0))
    dIh_thresh = np.array(dIh > 1, float)
    dIh_times = Vm_Ih_ts[np.ediff1d(dIh_thresh, to_begin=0) == 1]
    return dIh_times


# definition for finding start and stop times of brain states
def state_times(state_bool, ts, min_time, stitch_time):
    if np.sum(state_bool) == 0:
        state_start = np.empty(0)
        state_stop = np.empty(0)
    else:
        bool_diff = np.ediff1d(state_bool.astype(float), to_begin=0)
        state_start = ts[bool_diff == 1]
        state_stop = ts[bool_diff == -1]
        # remove runs that occur either at very start or very end of recording
        if state_start[0]-state_stop[0] > 0:
            state_stop = state_stop[1:]
        if state_start.shape != state_stop.shape:
            state_start = state_start[0:-1]
        # stitch together states that occur close together
        c = state_start[1:-1] - state_stop[0:-2]
        c = np.where(c < stitch_time)[0]
        state_start = np.delete(state_start, c+1)
        state_stop = np.delete(state_stop, c)
        # omit states that last less than the minimum time
        b = ((state_stop - state_start) > min_time)
        state_start = state_start[b]
        state_stop = state_stop[b]
    return state_start, state_stop


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

   
# %% load the data   

# set up list with an empty dictionary for each cell
data = [{'cell_id': 0, 'mouse_id': 0, 'synch': 0}
        for k in np.arange(cells.size)]


for i in np.arange(cells.size, dtype=int):
    # load the raw data from 1 cell at a time
    Vm_ts, Vm, lfp_ts, lfp, wh_ts, wh_speed, pupil_ts, pupil, Vm_Ih_ts, Vm_Ih = load_data_MIND(i)

    # find spike indices and make Vm trace with spikes removed
    thresh = 0.25
    refrac = 1.5
    search_period = 800
    sp_ind = find_sp_ind(Vm_ts, Vm, thresh, refrac)
    if sp_ind.size == 0:
        Vm_nosp = Vm
    else:
        Vm_nosp = remove_spikes(Vm_ts, Vm, sp_ind, search_period)
    
    # make the list of spike times if there are any
    sp_times = np.empty(0)
    if sp_ind.size > 0:
        sp_times = Vm_ts[sp_ind]

    # make smoothed Vm (after spikes removed)
    s_win = 40000  # number of data points
    Vm_nosp = pd.DataFrame(Vm_nosp)
    Vm_s = Vm_nosp.rolling(s_win).mean(center='true')
    
    # downsample Vm_s 3x to 20 Hz (factor of 1000)
    ds_factor = 10
    Vm_ds_ts_10, Vm_ds_10 = ds(Vm_ts, Vm_s, ds_factor)
    Vm_ds_ts_100, Vm_ds_100 = ds(Vm_ds_ts_10, Vm_ds_10, ds_factor)
    Vm_ds_ts, Vm_s_ds = ds(Vm_ds_ts_100, Vm_ds_100, ds_factor)
    
    # downsample Vm_nosp 3x to 20 Hz (factor of 1000)
    ds_factor = 10
    Vm_ds_ts_10, Vm_ds_10 = ds(Vm_ts, Vm_nosp, ds_factor)
    Vm_ds_ts_100, Vm_ds_100 = ds(Vm_ds_ts_10, Vm_ds_10, ds_factor)
    Vm_ds_ts, Vm_ds = ds(Vm_ds_ts_100, Vm_ds_100, ds_factor)
    
    # find the residual variance
    Vm_resid = Vm_nosp - Vm_s
    # calculate variance of Vm (calculated from residual)
    var_win = 20000
    Vm_resid = pd.DataFrame(Vm_resid)
    Vm_var = Vm_resid.rolling(var_win).var(center='true')
    # make smoothed variance
    s_win = 40000  # number of data points
    Vm_var = pd.DataFrame(Vm_var)
    Vm_var_s = Vm_var.rolling(s_win).mean(center='true')
    # downsample Vm_var 3x to 20 Hz (factor of 1000)
    ds_factor = 10
    Vm_ds_ts_10, Vm_var_ds_10 = ds(Vm_ts, Vm_var_s, ds_factor)
    Vm_ds_ts_100, Vm_var_ds_100 = ds(Vm_ds_ts_10, Vm_var_ds_10, ds_factor)
    Vm_ds_ts, Vm_var_ds = ds(Vm_ds_ts_100, Vm_var_ds_100, ds_factor)

    # find the running start and stop times
    run_start, run_stop = find_run_times(wh_ts, wh_speed)

    # add the z-scored spectrogram and theta index
    win = 2  # seconds
    overlap = 0.99  # 0-1, extent of window overlap
    frange = [0.5, 80]
    z_Sxx, f, spec_ts, t_d = prepare_Sxx(lfp_ts, lfp, win, overlap, frange)
    
    # find times where holding current is changed
    dIh_times = find_dIh_times(Vm_Ih_ts, Vm_Ih)

    # save relevant numbers in data list
    data[i]['cell_id'] = cells[i]
    data[i]['mouse_id'] = mouse_ids[i]
    data[i]['f'] = f
    data[i]['z_Sxx'] = z_Sxx
    data[i]['spec_ts'] = spec_ts
    data[i]['theta_delta'] = t_d
    data[i]['wh_speed'] = wh_speed
    data[i]['wh_ts'] = wh_ts
    data[i]['run_start'] = run_start
    data[i]['run_stop'] = run_stop
    data[i]['pupil_ts'] = pupil_ts
    data[i]['pupil'] = pupil
    data[i]['Vm_ds_ts'] = Vm_ds_ts
    data[i]['Vm_s_ds'] = Vm_s_ds
    data[i]['Vm_ds'] = Vm_ds
    data[i]['Vm_var'] = Vm_var_ds
    data[i]['lfp'] = lfp
    data[i]['lfp_ts'] = lfp_ts
    data[i]['sp_times'] = sp_times
    data[i]['dIh_times'] = dIh_times
    data[i]['Vm_Ih'] = Vm_Ih
    data[i]['Vm_Ih_ts'] = Vm_Ih_ts



# %% process the data
    
# extract spectral features from lfp, and assign brain states
for i in np.arange(len(data)):
    break_ind = np.arange(0, data[i]['spec_ts'].size, 5000)
    (d, t, b, g, total) = lfp_features(data[i]['spec_ts'], data[i]['f'],
                                       data[i]['z_Sxx'], break_ind)
    run_bool = find_running_bool(data[i]['spec_ts'], data[i]['run_start'],
                                 data[i]['run_stop'])
    theta_bool = (data[i]['theta_delta'] > 0)
    high_theta_bool = (data[i]['theta_delta'] > 1)
    SIA_bool = total < -0.2
    LIA_bool = total > 0
    high_LIA_bool = total > 0.25
    dial_bool = stats.zscore(data[i]['pupil']) > 0
    
    # find start and stop times for each brain state:
    # THETA
    min_time = 1
    stitch_time = 1
    g, h = state_times(theta_bool, data[i]['spec_ts'],
                       min_time, stitch_time)
    # of theta, keep only those periods that have really high theta
    j, k, l, m = find_run_theta(data[i]['spec_ts'], high_theta_bool,
                                g, h)
    data[i]['theta_start'] = j
    data[i]['theta_stop'] = k
    # divide theta into running and nonrunning-associated
    n, o, p, q = find_run_theta(data[i]['wh_ts'], data[i]['wh_speed'],
                                j, k)
    data[i]['run_theta_start'] = n
    data[i]['run_theta_stop'] = o
    data[i]['nonrun_theta_start'] = p
    data[i]['nonrun_theta_stop'] = q

    # PUPIL DILATIONS
    min_time = 1
    stitch_time = 1
    a, b = state_times(dial_bool, data[i]['pupil_ts'],
                       min_time, stitch_time)
    data[i]['dial_start'] = a
    data[i]['dial_stop'] = b
    # find dilations associated with run_theta and/or run
    run_theta_bool = find_running_bool(data[i]['spec_ts'],
                                       data[i]['run_theta_start'],
                                       data[i]['run_theta_stop'])
    c, d, e, f = find_run_theta(data[i]['spec_ts'],
                                (run_theta_bool + run_bool), a, b)
    data[i]['run_theta_dial_start'] = c
    data[i]['run_theta_dial_stop'] = d
    # find dilations associated with nonrun_theta
    nonrun_theta_bool = find_running_bool(data[i]['spec_ts'],
                                          data[i]['nonrun_theta_start'],
                                          data[i]['nonrun_theta_stop'])
    g, h, k, l = find_run_theta(data[i]['spec_ts'], nonrun_theta_bool, a, b)
    data[i]['nonrun_theta_dial_start'] = g
    data[i]['nonrun_theta_dial_stop'] = h
    
    # LIA
    min_time = 2
    stitch_time = 0.5
    c, d = state_times(LIA_bool, data[i]['spec_ts'],
                       min_time, stitch_time)
    # of LIA, keep only those periods that have really high total power
    e, f, g, h = find_run_theta(data[i]['spec_ts'], high_LIA_bool, c, d)
    # of those LIA periods, keep only those that do not overlap with theta/run
    temp_bool = (run_theta_bool + nonrun_theta_bool + run_bool)
    j, k, l, m = find_run_theta(data[i]['spec_ts'], temp_bool, e, f)
    data[i]['LIA_start'] = l
    data[i]['LIA_stop'] = m
    
    # find dilations associated with LIA
    LIA_bool = find_running_bool(data[i]['spec_ts'],
                                 data[i]['LIA_start'],
                                 data[i]['LIA_stop'])
    m, n, o, p = find_run_theta(data[i]['spec_ts'], LIA_bool, a, b)
    data[i]['LIA_dial_start'] = m
    data[i]['LIA_dial_stop'] = n
    # find dilations that occur outside LIA, run_theta, nonrun_theta
    temp_bool = (run_theta_bool + nonrun_theta_bool + LIA_bool +
                 run_bool)
    q, r, s, t = find_run_theta(data[i]['spec_ts'], temp_bool, a, b)
    data[i]['no_state_dial_start'] = s
    data[i]['no_state_dial_stop'] = t

states = [{'state':'theta', 'bef':-2.5, 'aft':0.5, 'samp_time':2, 't_win':[-3, 3]},
          {'state':'LIA', 'bef':-4, 'aft':-1, 'samp_time':2, 't_win':[-4, 2]},
          {'state':'run_theta', 'bef':-2.5, 'aft':0.5, 'samp_time':2, 't_win':[-3, 3]},
          {'state':'nonrun_theta', 'bef':-2.5, 'aft':0.5, 'samp_time':2, 't_win':[-3, 3]}]



# %% determine hyp/dep cells


# for each cell, determine whether it is significantly hyper or depolarizing
# bootstrap for the average Vm change at the start of theta
num_b = 1000
for l in np.arange(2):
    bef = states[l]['bef']
    aft = states[l]['aft']
    samp_time = states[l]['samp_time']
    state = states[l]['state']
    for i in np.arange(len(data)):
        samp_freq = 1/(data[i]['Vm_ds_ts'][1] - data[i]['Vm_ds_ts'][0])
        num_ind = int(samp_time*samp_freq)
        faux_d = np.zeros(num_b)
        # find index of dIh_times
        dIh_ind = data[i]['dIh_times']*samp_freq
        dIh_ind = dIh_ind.astype(int)
        for b in np.arange(num_b): 
            dVm = np.zeros(data[i][state+'_start'].size)
            for j in np.arange(data[i][state+'_start'].size):
                # find shifted indices, wrap around if necess., skip if straddling
                bef_ind = int(np.sum(data[i]['Vm_ds_ts'] <
                              (data[i][state+'_start'][j] + bef)) + b)
                aft_ind = int(np.sum(data[i]['Vm_ds_ts'] <
                              (data[i][state+'_start'][j] + aft)) + b)
                # wrap around times if beyond the end of recording
                while bef_ind >= data[i]['Vm_ds_ts'].size:
                    bef_ind = bef_ind - data[i]['Vm_ds_ts'].size
                    aft_ind = aft_ind - data[i]['Vm_ds_ts'].size
                # put nan if times are straddling a time when dIh is changed
                dIh_true = np.where((dIh_ind > bef_ind) &
                                    (dIh_ind < aft_ind + num_ind))[0]
                if dIh_true.size > 0:
                    dVm[j] = np.nan
                # put nan if times are straddling end of recording
                elif ((bef_ind < data[i]['Vm_ds_ts'].size) &
                     (aft_ind + num_ind > data[i]['Vm_ds_ts'].size)):
                    dVm[j] = np.nan
                else:
                    dVm[j] = (np.mean(data[i]['Vm_ds'][aft_ind:aft_ind+num_ind]) - 
                              np.mean(data[i]['Vm_ds'][bef_ind:bef_ind+num_ind]))
            faux_d[b] = np.nanmean(dVm)
        data[i][state+'_mean_dVm'] = faux_d[0]
        # calculate the p-value
        data[i][state+'_cell_p'] = np.sum(faux_d < faux_d[0])/num_b  




# %% save dictionaries using numpy
        

for i in np.arange(len(data)):
    np.save((r'C:\Users\Ashley\Documents\Ashley\Analysis\MIND\Python\Datasets\MIND-1\MIND1_v2\yesSxx\cell_' + 
             str(data[i]['cell_id']) + '.npy'), data[i]) 


# %% save dictionaries without Sxx
    
for i in np.arange(len(data)):
    del data[i]['z_Sxx']
    del data[i]['f']
    np.save((r'C:\Users\Ashley\Documents\Ashley\Analysis\MIND\Python\Datasets\MIND-1\MIND1_v2\noSxx\cell_' + 
             str(data[i]['cell_id']) + '.npy'), data[i])
    
