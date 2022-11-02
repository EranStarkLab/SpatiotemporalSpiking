import scipy.io as io
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from data_utils.light_removal import combine_list

from constants import SAMPLE_RATE, PV_COLOR, LIGHT_PV
MS_CONSTANT = 1000

def filter_pairs(pairs, types, amps, shanks, c_shank):
    """
    Filters out unwanted stimuli
    Parameters
    ----------
    pairs : Start and end times of the stimuli, only a specific duration is taken
    types : Type of the stimuli, only pulses are taken
    amps : Intensity of the stimulus, only strong stimuli are taken
    shanks : Shanks to which the stimulus was administrated, only same shank stimuli are taken
    c_shank : Shank of the unit

    Returns
    -------
    Filtered pairs
    """
    durs = np.array([b-a for a, b in pairs])
    mask = (types == 'PULSE') * (durs > 900) * (durs < 1100) * (amps > 0.059) * (shanks == c_shank)
    pairs = pairs[mask.astype('bool')]

    return pairs

def gaussian_func(sig, x):
    """

    Parameters
    ----------
    sig : sig : STD for the Gaussian distribution
    x : Inputs to the Gaussian function

    Returns
    -------
    Gaussian window
    """
    return np.exp(- (x ** 2) / (2 * sig ** 2))

def create_window(n, sig, sup):
    """

    Parameters
    ----------
    n : Length of the window
    sig : STD for the Gaussian distribution
    sup : Support value for the Gaussian distribution

    Returns
    -------

    """
    assert n - 1 == sig * sup * 2
    xs = np.linspace(-sup * sig, sup * sig + 1, n)
    wind = gaussian_func(sig, xs)
    return wind / wind.sum()

def mirror_edges(arr, wind_size):
    """

    Parameters
    ----------
    arr : Input array to mirror
    wind_size : How much to extend the arr with

    Returns
    -------
    New array with mirrored edges
    """
    left_edge = arr[1: wind_size // 2][::-1]
    right_edge = arr[-wind_size // 2 - 1: -1]

    return np.concatenate((left_edge, arr, right_edge), axis=0)

def calc_hist(spike_train, stims, bins, c):
    """

    Parameters
    ----------
    spike_train : List of spike times
    stims : List of triggers
    bins : Bin edges to use for the histogram

    Returns
    -------
    Peri-stimulus time histogram
    """
    spike_train = spike_train * SAMPLE_RATE / MS_CONSTANT
    wind_size = int(c * 3.5 * 2 + 1)
    g_wind = create_window(wind_size, c, 3.5)
    ret = np.zeros((len(stims), len(bins) - 1))
    for i, stim in enumerate(stims):
        ref_time_list = spike_train - stim
        mask = (ref_time_list >= bins[0]) * (ref_time_list < bins[-1])
        ref_time_list = ref_time_list[mask]
        hist, _ = np.histogram(ref_time_list, bins=bins) 
        mirr_hist = mirror_edges(hist, wind_size)
        conv_hist = np.convolve(mirr_hist, g_wind, mode='valid')
        ret[i] = conv_hist

    ret = ret * c * MS_CONSTANT

    return ret.mean(axis=0), stats.sem(ret, axis=0)


def main(cluster, data_path):
    fig, ax = plt.subplots(figsize=(30, 8))

    pairs_path = data_path + f'{cluster.filename}/{cluster.filename}.stm'
    mat = io.loadmat(pairs_path, simplify_cells=True)
    pairs, types, amps, shanks = mat['pairs'], mat['types'], mat['amps'], mat['shanks']
    pairs = filter_pairs(pairs * (SAMPLE_RATE // MS_CONSTANT), types, amps, shanks, cluster.shank)
    pairs = combine_list(pairs, margin=0)

    bins = np.linspace(-50 * SAMPLE_RATE // MS_CONSTANT, 100 * SAMPLE_RATE // MS_CONSTANT,
                       150 * SAMPLE_RATE // MS_CONSTANT + 1)

    spike_train = cluster.timings

    hist, sem = calc_hist(spike_train, [start for start, _ in pairs], bins, SAMPLE_RATE // MS_CONSTANT)

    rect = patches.Rectangle((0, 0), 50, (hist + sem).max(), facecolor='c', alpha=0.2)

    # Add the patch to the Axes
    ax.add_patch(rect)
    ax.plot(np.linspace(-50, 100, 150 * SAMPLE_RATE // MS_CONSTANT), hist, color=PV_COLOR)
    ax.fill_between(np.linspace(-50, 100, 150 * SAMPLE_RATE // MS_CONSTANT), hist - sem, hist + sem, color=LIGHT_PV, alpha=0.2)
    ax.set_ylabel('avg spike/second')
    ax.set_xlabel('ms from onset')

    return ax
