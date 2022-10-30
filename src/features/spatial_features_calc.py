import time
from enum import Enum

import numpy as np

from constants import VERBOS
from data_utils.clusters import Spike
from features.spatial_features.FET_spd import SPD
from features.spatial_features.FET_time_lag import TimeLagFeature
from features.spatial_features.FET_depolarization_graph import DepolarizationGraph

dep_spatial_features = [SPD()]
full_spatial_features = [TimeLagFeature(), DepolarizationGraph()]

class DELTA_MODE(Enum):
    """
    Enum to describe the different possible spatial events
    """
    NEG = 1
    F_MCROSS = 2
    S_MCROSS = 3


def calc_pos(arr, start_pos, mode):
    """

    Parameters
    ----------
    arr : List to find event in
    start_pos : Index to start looking from
    mode : Enum for the event

    Returns
    -------
    Index of the event occurrence
    """
    assert mode in [DELTA_MODE.F_MCROSS, DELTA_MODE.S_MCROSS]
    pos = start_pos
    while (pos >= 0) if mode == DELTA_MODE.F_MCROSS else (pos < len(arr)):
        if arr[pos] == 0:
            if mode == DELTA_MODE.F_MCROSS:
                pos -= 1
            else:
                pos += 1
        else:
            break
    return pos

def match_spike(channel, cons, med):
    """

    Parameters
    ----------
    channel : Mean waveform of the channel
    cons : Enum for the event
    med : Median used to determine median crossing

    Returns
    -------
    The transformed spike
    """
    pos = channel.argmin()
    spike = np.zeros(channel.shape)
    if cons == DELTA_MODE.NEG:
        spike[pos] = channel.min()
    elif cons in [DELTA_MODE.F_MCROSS, DELTA_MODE.S_MCROSS]:
        sig_m = np.convolve(np.where(channel <= med, -1, 1), [-1, 1], 'same')
        sig_m[0] = sig_m[-1] = 1
        pos = calc_pos(sig_m, pos, cons)
        spike[pos] = channel.min()
    else:
        raise KeyError("cons parameter is not valid")
    return spike, pos


def match_chunk(chunk, cons, amp):
    """

    Parameters
    ----------
    chunk : Mean waveform over all 8 channels
    cons : Enum for the event
    amp : Amplitudes of every channel of the chunk

    Returns
    -------
    Transformed chunk
    """
    ret = np.zeros(chunk.data.shape)
    main_c = amp.argmax()
    roll_val = 0
    med = np.median(chunk.data)
    for i, channel in enumerate(chunk.data):
        ret[i], pos = match_spike(channel, cons, med)
        if i == main_c:
            roll_val = chunk.data.shape[-1] // 2 - pos

    ret = np.roll(ret, roll_val, axis=1)
    chunk = Spike(data=ret / abs(ret.min()))
    
    return chunk

def wavelet_transform(chunks, cons, amps):
    """

    Parameters
    ----------
    chunks : Mean waveform for each chunk over all 8 channels
    cons : Enum for the event
    amps : Amplitudes of every channel for each chunk

    Returns
    -------
    List the same shape as chunks with the transformed spikes
    """
    ret = []
    for chunk, amp in zip(chunks, amps):
        ret.append(match_chunk(chunk, cons, amp))

    return ret


def calc_amps(chunks):
    """

    Parameters
    ----------
    chunks : Mean waveform for each chunk over all 8 channels

    Returns
    -------
    Amplitude of every channel in every chunk
    """
    ret = []
    for chunk in chunks:
        camps = chunk.data.max(axis=1) - chunk.data.min(axis=1)
        ret.append(camps / camps.max())

    return np.array(ret)

def calc_spatial_features(chunks):
    """

    Parameters
    ----------
    chunks : List of Spikes for which features are calculated

    Returns
    -------
    A matrix of spatial features for the chunks
    """
    feature_mat_for_cluster = None
    start_time = time.time()
    amps = calc_amps(chunks)

    wavelets_dep = wavelet_transform(chunks, DELTA_MODE.NEG, amps)
    wavelets_fzc = wavelet_transform(chunks, DELTA_MODE.F_MCROSS, amps)
    wavelets_szc = wavelet_transform(chunks, DELTA_MODE.S_MCROSS, amps)

    end_time = time.time()
    if VERBOS:
        print(f"wavelet transformation took {end_time - start_time:.3f} seconds")
    for feature in dep_spatial_features:
        start_time = time.time()
        mat_result = feature.calculate_feature(wavelets_dep)  # calculates the features, returns a matrix
        if feature_mat_for_cluster is None:
            feature_mat_for_cluster = mat_result
        else:
            feature_mat_for_cluster = np.concatenate((feature_mat_for_cluster, mat_result), axis=1)
        end_time = time.time()

        if VERBOS:
            print(f"feature {feature.name} contains {mat_result.shape} values")
            print(f"feature {feature.name} processing took {end_time - start_time:.3f} seconds")

    for feature in full_spatial_features:
        start_time = time.time()
        for data, dtype in zip([wavelets_dep, wavelets_fzc, wavelets_szc], ['dep', 'fzc', 'szc']):
            feature.set_data(dtype)
            mat_result = feature.calculate_feature(data, amps)  # calculates the features, returns a matrix
            if feature_mat_for_cluster is None:
                feature_mat_for_cluster = mat_result
            else:
                feature_mat_for_cluster = np.concatenate((feature_mat_for_cluster, mat_result), axis=1)
            end_time = time.time()

        if VERBOS:
            print(f"feature {feature.name} (data: {dtype}) contains {mat_result.shape} values")
            print(f"feature {feature.name} processing took {end_time - start_time:.3f} seconds")

    return feature_mat_for_cluster


def get_spatial_features_names():
    """

    Returns
    -------
    List of the names of the spatial features
    """
    names = []
    for feature in dep_spatial_features:
        names += feature.headers
    for feature in full_spatial_features:
        for dtype in ['NEG', 'FMC', 'SMC']:
            feature.set_data(dtype)
            names += feature.headers
    return names
