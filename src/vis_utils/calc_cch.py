import numpy as np
import matplotlib.pyplot as plt

CCH_RES = 1
CCH_WINDOW = 30
MS_CONSTANT = 1_000

def calc_hist(spike_train, stims, bins):
    """

    Parameters
    ----------
    spike_train : List of spike times
    stims : List of triggers
    bins : Bin edges to use for the histogram

    Returns
    -------
    Cross correlation histogram
    """
    ret = np.zeros(len(bins) - 1)
    for stim in stims:
        ref_time_list = spike_train - stim
        mask = (ref_time_list >= bins[0]) * (ref_time_list < bins[-1])
        ref_time_list = ref_time_list[mask]
        hist, _ = np.histogram(ref_time_list, bins=bins)
        ret += hist

    return MS_CONSTANT * ret / len(stims)


def main(pyr_clu, pv_clu):
    pyr_timings = pyr_clu.timings
    pv_timings = pv_clu.timings

    N = 2 * CCH_RES * CCH_WINDOW + 2  # times 2 for sides
    offset = 1 / (2 * 1)
    bins = np.linspace(-CCH_WINDOW - offset, CCH_WINDOW + offset, N)

    hist = calc_hist(pv_timings, pyr_timings, bins)
    fig, ax = plt.subplots()
    ax.vlines(0, ymin=0, ymax=hist.max(), color='k', linestyle='--')
    ax.bar(np.linspace(-CCH_WINDOW, CCH_WINDOW, N - 1), hist, color='k', width=bins[1] - bins[0])

    return ax
