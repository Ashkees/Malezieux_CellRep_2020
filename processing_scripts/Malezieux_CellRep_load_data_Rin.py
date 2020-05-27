# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 14:32:04 2019

@author: Ashley
"""

# Manuscript MIND-1
# Description: load Rin measurements and state detection


# %% load modules
import os
import numpy as np
import pandas as pd
from neo import io
from scipy import signal
from scipy import stats
from scipy.optimize import curve_fit



# %% load Data_to_Analyze_exHD.xlsx
sheets = ['Rin 1']
cells = np.empty(0)
mouse_ids = np.empty(0)
data_folder = np.empty(0)
ipdep_file = np.empty(0)
stepRa_file = np.empty(0)
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




# %% bunch of definitions

# definition for downsampling
def ds(ts, signal, ds_factor):
    signal_ds = np.mean(np.resize(signal,
                        (int(np.floor(signal.size/ds_factor)), ds_factor)), 1)
    ds_ts = ts[np.arange(int(np.round(ds_factor/2)), ts.size, ds_factor)]
    # trim off last time stamp if necessary
    ds_ts = ds_ts[0:signal_ds.size]
    return ds_ts, signal_ds

#a = r.read_protocol
#a[0].analogsignals[0].times
#a[0].analogsignals[0].data

# load data - LFP 200 Hz, running 32 Hz, raw Vm 20 kHz, Ih 20 kHz
# can handle both episodic and gap free
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
        sweep_start = int(good_seconds_start[i]/sweep_lenght[i]) 
        sweep_end = int((good_seconds_stop[i]/sweep_lenght[i]))
        sweep_pts = len(bl.segments[0].analogsignals[0].times)
        full_ts = np.zeros(sweep_pts*(sweep_end)) 
        for j in np.arange(sweep_end):
            start_ind = j*sweep_pts
            a = np.squeeze(bl.segments[j].analogsignals[0].times)
            full_ts[start_ind:start_ind+sweep_pts] = a
        Vm_ts = np.zeros(sweep_pts*(sweep_end-sweep_start)) 
        for j in np.arange(sweep_start, sweep_end):
            start_ind = j*sweep_pts - sweep_start*sweep_pts
            a = np.squeeze(bl.segments[j].analogsignals[0].times)
            Vm_ts[start_ind:start_ind+sweep_pts] = a
        lfp_raw = np.zeros(sweep_pts*(sweep_end-sweep_start))
        for k in np.arange(sweep_start, sweep_end):
            start_ind = k*sweep_pts - sweep_start*sweep_pts
            a = np.squeeze(bl.segments[k].analogsignals[1].data)
            lfp_raw[start_ind:start_ind+sweep_pts] = a
        Vm = np.zeros(sweep_pts*(sweep_end-sweep_start))
        for l in np.arange(sweep_start, sweep_end):
            start_ind = l*sweep_pts - sweep_start*sweep_pts
            a = np.squeeze(bl.segments[l].analogsignals[0].data)
            Vm[start_ind:start_ind+sweep_pts] = a
        
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
            TTL = np.zeros(sweep_pts*(sweep_end-sweep_start))
            for l in np.arange(sweep_start, sweep_end):
                start_ind = l*sweep_pts - sweep_start*sweep_pts
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
            TTL = np.zeros(sweep_pts*(sweep_end-sweep_start))
            for f in np.arange(sweep_start, sweep_end):
                start_ind = f*sweep_pts - sweep_start*sweep_pts
                a = np.squeeze(bl.segments[f].analogsignals[ind].data)
                TTL[start_ind:start_ind+sweep_pts] = a
            pupil_ts = Vm_ts[np.ediff1d(1*(TTL < 1), to_begin=0) > 0]
            file_name = os.path.join(data_folder[i], eye_track[i])
            pupil_excel = pd.read_excel(file_name)
            radii = pupil_excel.iloc[:, 0]
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
    ds_factor = 1
    if 'Chan2Hold' in channel_list:
        ind = channel_list.index('Chan2Hold')
        if np.isnan(sweep_lenght[i]):
           Vm_Ih = np.squeeze(np.copy(bl.segments[0].analogsignals[ind].data))
           Vm_Ih = Vm_Ih[(full_ts >= good_seconds_start[i]) &
                         (full_ts < good_seconds_stop[i])] 
        else:
            Vm_Ih = np.zeros(sweep_pts*(sweep_end-sweep_start))
            for f in np.arange(sweep_start, sweep_end):
                start_ind = f*sweep_pts - sweep_start*sweep_pts
                a = np.squeeze(bl.segments[f].analogsignals[ind].data)
                Vm_Ih[start_ind:start_ind+sweep_pts] = a
    else:
        Vm_Ih = np.empty(int(Vm_ts.size))


    return Vm_ts, Vm, lfp_ts, lfp, wh_ts, wh_speed, pupil_ts, pupil, Vm_Ih
    



# definition for finding run times
def find_run_times(wh_ts, wh_speed):
    nonzero_speed = np.array(wh_speed > 0, float)
    run_start = wh_ts[np.ediff1d(nonzero_speed, to_begin=0) == 1]
    run_stop = wh_ts[np.ediff1d(nonzero_speed, to_begin=0) == -1]
    # remove runs that occur either at very start or very end of recording
    if run_start.size > 0: 
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
    else:
        # keep just the first index per spike
        sp_ind = np.squeeze(np.where(np.diff(dVdt_thresh) == 1))
        # remove any duplicates of spikes that occur within refractory period
        samp_rate = 1/(Vm_ts[1]-Vm_ts[0])
        sp_ind = sp_ind[np.ediff1d(sp_ind,
                        to_begin=np.floor(samp_rate*refrac/1000+1)) >
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

# find the times of changes in Ih - both those for the Rin measurements,
# and long-term changes in overall applied current
# assumes there are many more Rin pulses than changes in holding current
# assumes all changes in Ih are in multiples of 10 pA
def find_dIh_times_Rin(Vm_Ih_ts, Vm_Ih):
    #Ih_s = pd.rolling_mean(Vm_Ih, 100)
    temp_Ih = pd.DataFrame(Vm_Ih)
    Ih_s = temp_Ih.rolling(100).mean(center='true')
    dIh = np.abs(np.ediff1d(Ih_s, to_begin=0))
    zIh = (dIh - np.nanmean(dIh))/np.nanstd(dIh)
    # replace nans with 0
    zIh[np.isnan(zIh)] = 0
    # find times of change in holding current
    # note: threshold of 0.5 makes some spurious dIh count twice, but I don't
    # think this will be a problem (it is worse when the threshold is 1 and some
    # dIh count too many times and confuse the algorithm)
    dIh_thresh = np.array(zIh > 0.5, float)
    all_dIh_times = Vm_Ih_ts[np.ediff1d(dIh_thresh, to_begin=0) == 1]
    # find the changes in current that occur at irregular times
    iei = np.round(np.ediff1d(all_dIh_times), decimals=3)
    unique_iei, counts = np.unique(iei, return_counts=True)
    # find the changes in Ih that occur at abnormal intervals
    dIh_mask = np.zeros(all_dIh_times.size, dtype=bool)
    for i in np.arange(unique_iei.size):
        if counts[i] == 1:
            dIh_mask[np.where(iei == unique_iei[i])] = True
    # remove the first instances of true in the mask (so we don't overcount dIh)
    dIh_mask[np.ediff1d(1*dIh_mask, to_begin=0) == 1] = False
    dIh_times = all_dIh_times[dIh_mask]
    # add a 0 to the beginning, so we can match with absolute current in each stage
    dIh_times = np.append(Vm_Ih_ts[0], dIh_times)       
    # now find the Rin measurements where there are no changes in overall Ih
    Rin_times = all_dIh_times[~dIh_mask]
    Rin_on = Rin_times[np.arange(0, Rin_times.size, 2)]
    Rin_off = Rin_times[np.arange(1, Rin_times.size, 2)]
    Rin_dur = np.round(np.mean(Rin_off - Rin_on), decimals=3)
    # remove any Rin measurements when there is a dIh within pulse_length of the Rin_on
    for i in np.arange(dIh_times.size):
        dt = np.abs(Rin_on - dIh_times[i])
        ind = np.where(dt < Rin_dur)
        Rin_on = np.delete(Rin_on, ind)
        Rin_off = np.delete(Rin_off, ind)
    # for each Rin measurement, record the change in current
    Rin_I_on = np.zeros(Rin_on.size)
    for i in np.arange(Rin_on.size):
        sample_I = Ih_s[np.logical_and(Vm_Ih_ts > Rin_on[i] - Rin_dur, Vm_Ih_ts < Rin_on[i])]
        Rin_I_on[i] = np.round(np.nanmean(sample_I), decimals=0)
    Rin_I_off = np.zeros(Rin_on.size)
    for i in np.arange(Rin_off.size):
        sample_I = Ih_s[np.logical_and(Vm_Ih_ts > Rin_off[i] - Rin_dur, Vm_Ih_ts < Rin_off[i])]
        Rin_I_off[i] = np.round(np.nanmean(sample_I), decimals=0)
    Rin_amp = np.round(Rin_I_off - Rin_I_on, decimals=-1)
    # record the absolute Rin_I_on between each change in overall holding current
    dIh_I = np.zeros(dIh_times.size)
    for i in np.arange(dIh_times.size):
        if i < dIh_times.size - 1:
            Rin_I_sample = Rin_I_on[np.logical_and(Rin_on >= dIh_times[i], Rin_on < dIh_times[i+1])]
            if Rin_I_sample.size == 0:
                I_sample = Vm_Ih[np.logical_and(Vm_Ih_ts >= dIh_times[i], Vm_Ih_ts < dIh_times[i+1])]
                dIh_I[i] = np.round(np.mean(I_sample), decimals=-1)
            else:
                dIh_I[i] = np.round(np.mean(Rin_I_sample), decimals=-1)
        else:
            Rin_I_sample = Rin_I_on[np.logical_and(Rin_on >= dIh_times[i], Rin_on < Vm_Ih_ts[-1])]
            if Rin_I_sample.size == 0:
                I_sample = Vm_Ih[np.logical_and(Vm_Ih_ts >= dIh_times[i], Vm_Ih_ts < Vm_Ih_ts[-1])]
                dIh_I[i] = np.round(np.mean(I_sample), decimals=-1)
            else:
                dIh_I[i] = np.round(np.mean(Rin_I_sample), decimals=-1)             
    return dIh_times, dIh_I, Rin_dur, Rin_amp, Rin_on, Rin_off


# measure voltage change and calculate Rin for each Rin measurement
# exclude measurements that contain spikes
def calc_Rin(Vm, Vm_ts, Rin_on, Rin_off, Rin_amp, samp_win, sp_times):
    # number of timestamps in samp_win
    samp_ind = np.searchsorted(Vm_ts-Vm_ts[0], samp_win)
    # take mean voltage just before Rin_on and Rin_off times
    # exclude times with spikes
    Rin_on_V = np.zeros(Rin_on.size)
    Rin_off_V = np.zeros(Rin_on.size)
    for i in np.arange(Rin_on.size):
        if np.any(np.logical_and(sp_times >= Rin_on[i]-2*samp_win, sp_times <= Rin_on[i]+samp_win)):
            Rin_on_V[i] = np.nan
            Rin_off_V[i] = np.nan
        elif np.any(np.logical_and(sp_times >= Rin_off[i]-2*samp_win, sp_times <= Rin_off[i]+samp_win)):
            Rin_on_V[i] = np.nan
            Rin_off_V[i] = np.nan
        else:
            on_ind = np.searchsorted(Vm_ts, Rin_on[i])
            off_ind = np.searchsorted(Vm_ts, Rin_off[i])
            Rin_on_V[i] = np.nanmean(Vm[on_ind-samp_ind : on_ind])
            Rin_off_V[i] = np.nanmean(Vm[off_ind-samp_ind : off_ind])
    dV = Rin_off_V - Rin_on_V
    Rin = 1000*dV/Rin_amp
    return Rin
    
# definition to make a boolean to determine which timestamps (or Rin measurements)
# occur in a state
def find_state_bool(Rin_on, state_start, state_stop):
    state_bool = np.zeros(Rin_on.size)
    for i in np.arange(state_start.size):
        ind_start = np.searchsorted(Rin_on, state_start[i])
        ind_stop = np.searchsorted(Rin_on, state_stop[i])
        state_bool[ind_start:ind_stop] = np.ones(ind_stop-ind_start)
    return state_bool
        

# load data for currinj protocol - raw Vm 20 kHz and current injection trace
# episodic only
def load_data_MIND_ipdep(cell_ind):

    i = cell_ind

    # load some Axon data from ABF files
    file_name = os.path.join(data_folder[i], ipdep_file[i])
    # r is the name bound to the object created by io.AxonIO
    r = io.AxonIO(filename=file_name)
    # bl is the object that actually has the data, created by read_block
    bl = r.read_block()

    # get list of channel names
    channel_list = []
    for asig in bl.segments[0].analogsignals:
        channel_list.append(asig.name)
    
    sweep_start = 0
    sweep_end = len(bl.segments)
    sweep_pts = len(bl.segments[0].analogsignals[0].times)
    full_ts = np.zeros([sweep_pts, sweep_end]) 
    for j in np.arange(sweep_end):
        full_ts[:, j] = np.squeeze(bl.segments[j].analogsignals[0].times)
    ipdep_ts = np.zeros([sweep_pts, (sweep_end-sweep_start)]) 
    for j in np.arange(sweep_start, sweep_end):
        ipdep_ts[:, j-sweep_start] = np.squeeze(bl.segments[j].analogsignals[0].times)
    ipdep_Vm = np.zeros([sweep_pts, (sweep_end-sweep_start)])
    for l in np.arange(sweep_start, sweep_end):
        ipdep_Vm[:, l-sweep_start] = np.squeeze(bl.segments[l].analogsignals[0].data)
    ind = channel_list.index('Chan2Hold')
    ipdep_I = np.zeros([sweep_pts, (sweep_end-sweep_start)])
    for f in np.arange(sweep_start, sweep_end):
        ipdep_I[:, f-sweep_start] = np.squeeze(bl.segments[f].analogsignals[ind].data)

    return ipdep_ts, ipdep_Vm, ipdep_I



# load data for step Ra protocol - raw Vm 20 kHz and voltage injection trace
# episodic only
def load_data_MIND_stepRa(cell_ind):

    i = cell_ind

    # load some Axon data from ABF files
    file_name = os.path.join(data_folder[i], stepRa_file[i])
    # r is the name bound to the object created by io.AxonIO
    r = io.AxonIO(filename=file_name)
    # bl is the object that actually has the data, created by read_block
    bl = r.read_block()

    # get list of channel names
    channel_list = []
    for asig in bl.segments[0].analogsignals:
        channel_list.append(asig.name)
    
    sweep_start = 0
    sweep_end = len(bl.segments)
    sweep_pts = len(bl.segments[0].analogsignals[0].times)
    full_ts = np.zeros([sweep_pts, sweep_end]) 
    for j in np.arange(sweep_end):
        full_ts[:, j] = np.squeeze(bl.segments[j].analogsignals[0].times)
    stepRa_ts = np.zeros([sweep_pts, (sweep_end-sweep_start)]) 
    for j in np.arange(sweep_start, sweep_end):
        stepRa_ts[:, j-sweep_start] = np.squeeze(bl.segments[j].analogsignals[0].times)
    ind = channel_list.index('Chan2')
    stepRa_Im = np.zeros([sweep_pts, (sweep_end-sweep_start)])
    for l in np.arange(sweep_start, sweep_end):
        stepRa_Im[:, l-sweep_start] = np.squeeze(bl.segments[l].analogsignals[ind].data)
    ind = channel_list.index('Chan2Hold')
    stepRa_V = np.zeros([sweep_pts, (sweep_end-sweep_start)])
    for f in np.arange(sweep_start, sweep_end):
        stepRa_V[:, f-sweep_start] = np.squeeze(bl.segments[f].analogsignals[ind].data)

    return stepRa_ts, stepRa_Im, stepRa_V



def single_exp(x, a, b):
    return a * np.exp(b * x)

def double_exp(x, a, b, c, d, f):
    return a * np.exp(b * x) + c * np.exp(d * x) + f
    
def linear(x, m, b):
    return m*x + b

def linear_origin(x, m):
    return m*x

# definition for using ipdep traces to find the IV curve, and exponential fitting
# to the onset and offset of the pulse
def analyze_ipdep(ipdep_ts, ipdep_Vm, ipdep_I, ss_win, fit_win, dp):
    
    deltapoint = sum(ipdep_ts[:,0] < ipdep_ts[0,0] + dp)
    ss_win_ind = sum(ipdep_ts[:,0] < ipdep_ts[0,0] + ss_win)
    fit_ind = sum(ipdep_ts[:,0] < ipdep_ts[0,0] + fit_win)
    
    # find the pulse start and end
    I_diff = np.ediff1d(ipdep_I[:, 0], to_begin=0)
    pulse_ind = np.array([np.argmin(I_diff), np.argmax(I_diff)])
    pulse_ind = np.sort(pulse_ind)
    pulse_time = ipdep_ts[pulse_ind, 0]
    
    # find which traces do not have spikes during the pulse
    thresh = -10
    sweeps_no_sp = np.sum(ipdep_Vm[pulse_ind[0]-ss_win_ind:pulse_ind[1]+ss_win_ind] > thresh,
                          axis=0) == 0
    
    # find the traces that do not saturate for more than 100 points
    sweeps_no_sat = np.sum(np.round(ipdep_Vm, decimals=1) == -100, axis=0) < 100
    
    good_sweeps = sweeps_no_sp*sweeps_no_sat
    
    # calculate the delta I and delta ss_V for the good sweeps
    base_ind = [0, ss_win_ind]
    ss_ind = [pulse_ind[1] - ss_win_ind, pulse_ind[1]]
    base_V = np.nanmean(ipdep_Vm[base_ind[0]:base_ind[1], :], axis=0)
    ss_V = np.nanmean(ipdep_Vm[ss_ind[0]:ss_ind[1], :], axis=0)
    delta_V = ss_V - base_V
    delta_V[~good_sweeps] = np.nan
    
    # calculate the magnitude of the current pulses
    delta_I = (np.nanmean(ipdep_I[ss_ind[0]:ss_ind[1], :], axis=0) - 
               np.nanmean(ipdep_I[base_ind[0]:base_ind[1], :], axis=0))
    delta_I = np.round(delta_I, decimals=-1)
    delta_I[~good_sweeps] = np.nan
    
    
    # calculate tau at onset and offset of each pulse
    p0 = [10, -1000, 1, -30, 0]
    exp_ts = ipdep_ts[0:fit_ind - deltapoint, 0] - ipdep_ts[0, 0]
    exp_traces = np.zeros([fit_ind - deltapoint, delta_I.size , 2])
    popt = np.zeros([5, delta_I.size, 2])
    tau = np.zeros([2, delta_I.size, 2])
    for j in np.arange(delta_I.size):
        if good_sweeps[j]:  # skip the saturating traces
            # only analyze traces that have a nonzero pulse (more than 10 mV)
            if np.abs(delta_I[j]) > 10:
                Vm = ipdep_Vm[:, j]
                # isolate the exp part of the trace, fit a double exp, find tau
                # repeat for beginning and end of pulse
                for p in np.arange(2):
                    if p == 0:
                        flip = -1
                    if p == 1:
                        flip = 1
                    trace = flip*np.sign(delta_I[j])*Vm[pulse_ind[p] + deltapoint : pulse_ind[p] + fit_ind]
                    trace = trace-trace[-1]
                    exp_traces[:, j, p] = trace
                    
                    # calculate tau by fitting a double exponential
                    try:
                        temp_popt, pcov = curve_fit(double_exp, exp_ts, trace, p0)
                        tau[:, j, p] = np.sort([-1/temp_popt[1], -1/temp_popt[3]])
                        popt[:, j, p] = temp_popt
                    except RuntimeError:
                        popt[:, j, p] = np.full([5], np.nan)
                        tau[:, j, p] = np.full([2], np.nan)
                    except ValueError:
                        popt[:, j, p] = np.full([5], np.nan)
                        tau[:, j, p] = np.full([2], np.nan)
            # put nan when there is a pulse of 0 pA
            else:
                exp_traces[:, j, :] = np.full([fit_ind - deltapoint, 2], np.nan)
                popt[:, j, :] = np.full([5, 2], np.nan)
                tau[:, j, :] = np.full([2, 2], np.nan)
        else:
            exp_traces[:, j, :] = np.full([fit_ind - deltapoint, 2], np.nan)
            popt[:, j, :] = np.full([5, 2], np.nan)
            tau[:, j, :] = np.full([2, 2], np.nan)
    return pulse_time, delta_I, delta_V, exp_traces, exp_ts, popt, tau


def analyze_stepRa(stepRa_ts, stepRa_Im, stepRa_V, ss_win):
    
    ss_win_ind = sum(stepRa_ts[:,0] < stepRa_ts[0,0] + ss_win)
    
    # find the pulse start and end
    V_diff = np.ediff1d(stepRa_V[:, 0], to_begin=0)
    pulse_ind = np.array([np.argmin(V_diff), np.argmax(V_diff)])
    pulse_ind = np.sort(pulse_ind)
    pulse_time = stepRa_ts[pulse_ind, 0]
    
    # calculate the delta ss_I and delta V for the average traces
    avg_Im = np.nanmean(stepRa_Im, axis=1)
    base_ind = [0, ss_win_ind]
    ss_ind = [pulse_ind[1] - ss_win_ind, pulse_ind[1]]
    base_Im = np.nanmean(avg_Im[base_ind[0]:base_ind[1]], axis=0)
    ss_Im = np.nanmean(avg_Im[ss_ind[0]:ss_ind[1]], axis=0)
    delta_Im = ss_Im - base_Im
    
    # calculate the magnitude of the current pulses
    avg_V = np.nanmean(stepRa_V, axis=1)
    delta_V = (np.nanmean(avg_V[ss_ind[0]:ss_ind[1]], axis=0) - 
               np.nanmean(avg_V[base_ind[0]:base_ind[1]], axis=0))
    delta_V = np.round(delta_V, decimals=0)
    
    # calculate the change in max current
    if delta_V < 0:
        max_Im = np.nanmax(avg_Im)
        min_Im = np.nanmin(avg_Im)
        delta_maxIm = np.mean([base_Im - min_Im, max_Im - ss_Im])
    else:
        max_Im = np.nanmax(avg_Im)
        min_Im = np.nanmin(avg_Im)
        delta_maxIm = np.mean([ss_Im - min_Im, max_Im - base_Im])

    #calculate the resistances
    stepRa_Rin = 1000*delta_V/delta_Im
    Ra = np.abs(1000*delta_V/delta_maxIm)
    
    return pulse_time, stepRa_Rin, Ra


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



   
# %% do something with the data   

# set up list with an empty dictionary for each cell
data = [{'cell_id': 0, 'mouse_id': 0, 'synch': 0}
        for k in np.arange(cells.size)]

#
for i in np.arange(cells.size, dtype=int):
    # load the raw data from 1 cell at a time
    Vm_ts, Vm, lfp_ts, lfp, wh_ts, wh_speed, pupil_ts, pupil, Vm_Ih = load_data_MIND(i)
    
    # find times where holding current is changed
    # and where Rin measurements are taken
    dIh_times, dIh_I, Rin_dur, Rin_amp, Rin_on, Rin_off = find_dIh_times_Rin(Vm_ts, Vm_Ih)
    
    # find spike indices and make Vm trace with spikes removed
#    thresh = 0.25 #regular threshold
    thresh = 1.2 #For Rin analysis, threshold changed because it was detecting the Vm changes as spikes
    refrac = 1.5
    search_period = 800
    sp_ind = find_sp_ind(Vm_ts, Vm, thresh, refrac)
#    if sp_ind.size == 0:
#        Vm_nosp = Vm
#    else:
#        Vm_nosp = remove_spikes(Vm_ts, Vm, sp_ind, search_period)
    
    # make the list of spike times if there are any
    sp_times = np.empty(0)
    if sp_ind.size > 0:
        sp_times = Vm_ts[sp_ind]

    # find the running start and stop times
    run_start, run_stop = find_run_times(wh_ts, wh_speed)

    # add the z-scored spectrogram and theta index
    win = 2  # seconds
    overlap = 0.99  # 0-1, extent of window overlap
    frange = [0.5, 80]
    z_Sxx, f, spec_ts, t_d = prepare_Sxx(lfp_ts, lfp, win, overlap, frange)
    
    # extract spectral features from lfp, and assign brain states
    break_ind = np.arange(0, spec_ts.size, 5000)
    (d, t, b, g, total) = lfp_features(spec_ts, f,
                                       z_Sxx, break_ind)
    run_bool = find_running_bool(spec_ts, run_start,
                                 run_stop)
    theta_bool = (t_d > 0)
    high_theta_bool = (t_d > 1)
    SIA_bool = total < -0.2
    LIA_bool = total > 0
    high_LIA_bool = total > 0.25
    dial_bool = stats.zscore(pupil) > 0
    high_LIA_bool = high_LIA_bool
    total = total
    
    # find start and stop times for each brain state:
    # THETA
    min_time = 1
    stitch_time = 1
    g, h = state_times(theta_bool, spec_ts,
                       min_time, stitch_time)
    # of theta, keep only those periods that have really high theta
    j, k, l, m = find_run_theta(spec_ts, high_theta_bool,
                                g, h)
    data[i]['theta_start'] = j
    data[i]['theta_stop'] = k
    # divide theta into running and nonrunning-associated
    n, o, p, q = find_run_theta(wh_ts, wh_speed,
                                j, k)
    data[i]['run_theta_start'] = n
    data[i]['run_theta_stop'] = o
    data[i]['nonrun_theta_start'] = p
    data[i]['nonrun_theta_stop'] = q

    # PUPIL DILATIONS
    min_time = 1
    stitch_time = 1
    a, b = state_times(dial_bool, pupil_ts,
                       min_time, stitch_time)
    data[i]['dial_start'] = a
    data[i]['dial_stop'] = b
    # find dilations associated with run_theta and/or run
    run_theta_bool = find_running_bool(spec_ts,
                                       data[i]['run_theta_start'],
                                       data[i]['run_theta_stop'])
    c, d, e, f = find_run_theta(spec_ts,
                                (run_theta_bool + run_bool), a, b)
    data[i]['run_theta_dial_start'] = c
    data[i]['run_theta_dial_stop'] = d
    # find dilations associated with nonrun_theta
    nonrun_theta_bool = find_running_bool(spec_ts,
                                          data[i]['nonrun_theta_start'],
                                          data[i]['nonrun_theta_stop'])
    g, h, k, l = find_run_theta(spec_ts, nonrun_theta_bool, a, b)
    data[i]['nonrun_theta_dial_start'] = g
    data[i]['nonrun_theta_dial_stop'] = h
    
    # LIA
    min_time = 2
    stitch_time = 0.5
    c, d = state_times(LIA_bool, spec_ts,
                       min_time, stitch_time)
    # of LIA, keep only those periods that have really high total power
    e, f, g, h = find_run_theta(spec_ts, high_LIA_bool, c, d)
    # of those LIA periods, keep only those that do not overlap with theta/run
    temp_bool = (run_theta_bool + nonrun_theta_bool + run_bool)
    j, k, l, m = find_run_theta(spec_ts, temp_bool, e, f)
    data[i]['LIA_start'] = l
    data[i]['LIA_stop'] = m
    
    # find dilations associated with LIA
    LIA_bool = find_running_bool(spec_ts,
                                 data[i]['LIA_start'],
                                 data[i]['LIA_stop'])
    m, n, o, p = find_run_theta(spec_ts, LIA_bool, a, b)
    data[i]['LIA_dial_start'] = m
    data[i]['LIA_dial_stop'] = n
    # find dilations that occur outside LIA, run_theta, nonrun_theta
    temp_bool = (run_theta_bool + nonrun_theta_bool + LIA_bool +
                 run_bool)
    q, r, s, t = find_run_theta(spec_ts, temp_bool, a, b)
    data[i]['no_state_dial_start'] = s
    data[i]['no_state_dial_stop'] = t
 
    # save relevant numbers in data list
    data[i]['cell_id'] = cells[i]
    data[i]['mouse_id'] = mouse_ids[i]
    #data[i]['f'] = f
    #data[i]['z_Sxx'] = z_Sxx
    #data[i]['spec_ts'] = spec_ts
    #data[i]['theta_delta'] = t_d
    #data[i]['wh_speed'] = wh_speed
    #data[i]['wh_ts'] = wh_ts
    #data[i]['pupil_ts'] = pupil_ts
    #data[i]['pupil'] = pupil
    data[i]['run_start'] = run_start
    data[i]['run_stop'] = run_stop
    data[i]['Vm_ts'] = Vm_ts
    data[i]['Vm'] = Vm
    #data[i]['Vm_nosp'] = Vm_nosp
    #data[i]['lfp'] = lfp
    #data[i]['lfp_ts'] = lfp_ts
    data[i]['sp_times'] = sp_times
    data[i]['dIh_times'] = dIh_times
    data[i]['dIh_I'] = dIh_I
    #data[i]['Vm_Ih'] = Vm_Ih
    data[i]['Rin_dur'] = Rin_dur
    data[i]['Rin_amp'] = Rin_amp
    data[i]['Rin_on'] = Rin_on
    data[i]['Rin_off'] = Rin_off


# load and analyze stepRa files
for i in np.arange(cells.size, dtype=int):
    
    if isinstance(stepRa_file[i], str):
        
        # load the stepRa traces from one cell and extract relevant information
       stepRa_ts, stepRa_Im, stepRa_V = load_data_MIND_stepRa(i)
       
       ss_win = 0.1  # seconds over which to calcuate the steady state
       pulse_time, stepRa_Rin, Ra = analyze_stepRa(stepRa_ts, stepRa_Im, stepRa_V, ss_win)
       
       data[i]['stepRa_ts'] = stepRa_ts
       data[i]['stepRa_Im'] = stepRa_Im
       data[i]['stepRa_V'] = stepRa_V
       data[i]['stepRa_Rin'] = stepRa_Rin
       data[i]['Ra'] = Ra
       
    else:
       data[i]['stepRa_ts'] = np.nan
       data[i]['stepRa_Im'] = np.nan
       data[i]['stepRa_V'] = np.nan
       data[i]['stepRa_Rin'] = np.nan
       data[i]['Ra'] = np.nan


# load and analyze ipdep files
for i in np.arange(cells.size, dtype=int):
    
    if isinstance(ipdep_file[i], str):
    
        # load the ipdep traces from one cell and extract relevant information
        ipdep_ts, ipdep_Vm, ipdep_I = load_data_MIND_ipdep(i)
        
        # extract the relevant information from the ipdep traces
        ss_win = 0.05  # in seconds, window to calculate steady state
        fit_win = 0.1  # in seconds, window to calculate the exponential fit
        dp = 0  # in seconds, amt of trace to disregard after pulse start/end
        a, b, c, d, e, f, g = analyze_ipdep(ipdep_ts, ipdep_Vm, ipdep_I, ss_win, fit_win, dp)
        pulse_time = a
        delta_I = b
        delta_V = c
        exp_traces = d
        exp_ts = e
        popt = f
        tau = g
        
        # calculate Rin from linear fit of IV curve - units are MOhms
        delta_V = delta_V[~np.isnan(delta_I)]
        delta_I = delta_I[~np.isnan(delta_I)]
        popt, pcov = curve_fit(linear, delta_I, delta_V)
        data[i]['ipdep_Rin'] = 1000*popt[0]
        # membrane time constant (tau) for the cell is the median of taus - units is seconds
        data[i]['ipdep_tau'] = np.nanmedian(tau[1, :, :])
        # leak conductance is the inverse of the Rin (bad estimate because the cell
        # is not at rest - it is an upper bound) - units are nS
        data[i]['ipdep_g_l'] = 1000/data[i]['ipdep_Rin']
        # capacitance is the product of tau and g leak - units are nF
        data[i]['ipdep_C'] = data[i]['ipdep_tau']*data[i]['ipdep_g_l']
        # find cell surface area, as C = 1 µF/cm^2 - units are cm^2
        data[i]['ipdep_sa'] = data[i]['ipdep_C']/1000
        
    else:
        data[i]['ipdep_Rin'] = np.nan
        data[i]['ipdep_tau'] = np.nan
        data[i]['ipdep_g_l'] = np.nan
        data[i]['ipdep_C'] = np.nan
        data[i]['ipdep_sa'] = np.nan




# calculate input resistances
for i in np.arange(len(data)):
    samp_win = 0.25*data[i]['Rin_dur']
    Rin = calc_Rin(data[i]['Vm'], data[i]['Vm_ts'], data[i]['Rin_on'],
                   data[i]['Rin_off'], data[i]['Rin_amp'], samp_win,
                   data[i]['sp_times'])
    data[i]['Rin'] = Rin

## find % of invalid Rin measurements in each cell
#num_Rin_invalid = np.array([np.sum(np.isnan(d['Rin'])) for d in data])
#num_all_Rin = np.array([d['Rin'].size for d in data])
#perc_Rin_invalid = num_Rin_invalid/num_all_Rin
## plot the problem Vm traces to see if any Rin can be saved
#i = 1
#plt.figure()
#plt.plot(data[i]['Vm_ts'], data[i]['Vm'])
#plt.scatter(data[i]['Rin_on'][np.isnan(data[i]['Rin'])],
#            -30*np.ones(np.sum(np.isnan(data[i]['Rin']))), color='r')
## find number of negative Rin measurements for each cell (should be few)
#[np.sum(d['Rin']<0) for d in data]

# find the Rin measurements that are during each state, and during no state
state = ['theta', 'LIA'] 
for i in np.arange(len(data)):
    nost_bool = np.zeros(data[i]['Rin_on'].size)
    for l in np.arange(len(state)):
        state_bool = find_state_bool(data[i]['Rin_on'], data[i][state[l]+'_start'],
                                     data[i][state[l]+'_stop'])
        data[i][state[l]+'_Rin'] = [state_bool == 1]
        nost_bool = nost_bool + state_bool
    data[i]['nost_Rin'] = [nost_bool == 0]



# %% delete unused keys

keys2delete = ['dial_start', 'dial_stop', 'no_state_dial_start',
               'no_state_dial_stop', 'LIA_dial_start', 'LIA_dial_stop',
               'run_theta_dial_start', 'run_theta_dial_stop',
               'nonrun_theta_dial_start', 'nonrun_theta_dial_stop', 'synch',
               'Vm_ts', 'Vm', 'stepRa_ts', 'stepRa_Im', 'stepRa_V']

for i in np.arange(len(data)):
    for j in np.arange(len(keys2delete)):
        if keys2delete[j] in data[i].keys():
            del data[i][keys2delete[j]]



# %% save into numpy files

# Save dictionaries using numpy
for i in np.arange(len(data)):
    np.save((r'C:\Users\akees\Documents\Ashley\Papers\MIND 1\Cell Reports\Dryad upload\Dataset_Rin\cell_' + 
             str(data[i]['cell_id']) + '.npy'), data[i]) 
