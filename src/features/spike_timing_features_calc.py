import numpy as np
import scipy.signal as signal
import time
import warnings
from constants import VERBOS, ACH_WINDOW, ACH_RESOLUTION, UPSAMPLE

from features.spike_timing_features.FET_DKL import DKL
from features.spike_timing_features.FET_jump_index import Jump
from features.spike_timing_features.FET_PSD import PSD
from features.spike_timing_features.FET_rise_time import RiseTime
from features.spike_timing_features.FET_unif_dist import UnifDist
from features.spike_timing_features.FET_firing_rate import FiringRate

features = [DKL(), Jump(), PSD(), RiseTime(), UnifDist()]
ind_features = [FiringRate()]


def get_array_length(chunks):
    """
    
    Parameters
    ----------
    chunks : List of spike indices related to each chunk

    Returns
    -------
    Total number of spikes partitioned into chunks
    """
    counter = 0
    for chunk in chunks:
        counter += len(chunk)
    return counter


def invert_chunks(chunks):
    """

    Parameters
    ----------
    chunks : List of spike indices related to each chunk

    Returns
    -------
    List the length of the number of spikes, with the values represent chnk index
    """
    ret = np.zeros(get_array_length(chunks), dtype=np.int)
    for i, chunk in enumerate(chunks):
        ret[chunk] = i  # note that chunk is a list
    return ret


def ordered_array_histogram(lst, bins, i0):
    """
    
    Parameters
    ----------
    lst : List of Spike-timing
    bins : Ordered List describing the edges of the bins in the ACH
    i0 : Integer; index of trigger spike

    Returns
    -------

    """
    max_abs = bins[-1]
    bin_size = bins[1] - bins[0]
    bin0 = len(bins) // 2
    hist = np.zeros(len(bins) - 1)

    i = i0 + 1
    while i <= len(lst) - 1 and lst[i] < max_abs:
        bin_ind = int(((lst[i] - (bin_size / 2)) // bin_size) + bin0)
        hist[bin_ind] += 1
        i += 1
    i = i0 - 1
    while i >= 0 and lst[i] >= -max_abs:
        bin_ind = int(((lst[i] - (bin_size / 2)) // bin_size) + bin0)
        hist[bin_ind] += 1
        i -= 1

    return hist

def calc_temporal_histogram(time_lst, bins, chunks):
    """

    Parameters
    ----------
    time_lst : List of the timing (in ms) of the spikes of the unit
    bins : Ordered List describing the edges of the bins in the ACH
    chunks : List of spike indices related to each chunk

    Returns
    -------
    List of ACHs
    """
    ret = np.zeros((len(chunks), len(bins) - 1))
    counter = np.zeros((len(chunks), 1))
    chunks_inv = invert_chunks(chunks)
    for i in range(len(time_lst)):

        ref_time_list = time_lst - time_lst[i]
        hist = ordered_array_histogram(ref_time_list, bins, i)
        ret[chunks_inv[i]] += hist
        counter[chunks_inv[i]] += 1

    ret = ret / (counter * ((bins[1] - bins[0])/1000))

    return ret


def calc_st_features(time_lst, chunks, resolution=ACH_RESOLUTION, upsample=UPSAMPLE, start_band_range=50,
                     ach_range=ACH_WINDOW):
    """
    Parameters
    ----------
    time_lst : List of the timing (in ms) of the spikes of the unit
    chunks : List of spike indices related to each chunk
    resolution : ACH resolution (bins per ms)
    upsample : Upsampling factor for the ACH
    start_band_range : Number of ms in the initial ACH of the unit
    ach_range : Number of ms in the entire ACH

    Returns
    -------
    A matrix of spike-timing features for the chunks
    """
    time_lst = np.array(time_lst)

    assert (resolution > 0 and resolution % 1 == 0)
    assert (ach_range > 0 and ach_range % 1 == 0)

    N = 2 * resolution * ach_range + 2
    offset = 1 / (2 * resolution)
    bins = np.linspace(-ach_range - offset, ach_range + offset, N)
    start_time = time.time()
    histograms = calc_temporal_histogram(time_lst, bins, chunks)
    zero_bin_ind = histograms.shape[1] // 2
    histograms = (histograms[:, :zero_bin_ind + 1:][:, ::-1] + histograms[:, zero_bin_ind:]) / 2

    histograms = np.array([signal.resample_poly(histogram, upsample ** 2, upsample, padtype='line') for histogram in histograms])
    histograms = np.where(histograms >= 0, histograms, 0)
    end_time = time.time()
    if VERBOS:
        print(f"histogram creation took {end_time - start_time:.3f} seconds")
    start_band = histograms[:, :resolution * start_band_range * upsample]
    mid_band = histograms[:, resolution * start_band_range * upsample: resolution * ach_range * upsample + 1]

    feature_mat_for_cluster = None
    assert len(ind_features) == 1
    for feature in ind_features:
        feature_mat_for_cluster = feature.calculate_feature(time_lst, chunks)

    for feature in features:
        start_time = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mat_result = feature.calculate_feature(start_band=start_band, mid_band=mid_band, rhs=histograms)

        feature_mat_for_cluster = np.concatenate((feature_mat_for_cluster, mat_result), axis=1)
        end_time = time.time()
        if VERBOS:
            print(f"feature {feature.name} processing took {end_time - start_time:.3f} seconds")
    mat_result = np.ones((len(chunks), 1)) * len(time_lst)
    feature_mat_for_cluster = np.concatenate((feature_mat_for_cluster, mat_result), axis=1)
    return feature_mat_for_cluster


def get_st_features_names():
    """
    
    Returns
    -------
    List of the names of the spike-timing features
    """
    names = []
    for feature in ind_features:
        names += feature.headers
    for feature in features:
        names += feature.headers
    names += ['num_spikes']
    return names
