import scipy.io as io
import numpy as np

def get_idnds(time_lst, pairs, remove_lights):
    """

    Parameters
    ----------
    time_lst : Iterable of times of spikes in ms
    pairs : start and end times of the light stimuli
    remove_lights : Bool; remove spikes during stimuli or remove spikes not during stimuli

    Returns
    -------
    Iterable of integers; valid indices
    """
    i, j = 0, 0
    inds = []

    while i < len(time_lst):
        if j >= len(pairs):
            if remove_lights:
                inds.append(i)
            i += 1
        elif pairs[j][1] < time_lst[i]:
            j += 1
        elif pairs[j][0] <= time_lst[i] <= pairs[j][1]:
            if not remove_lights:
                inds.append(i)
            i += 1
        else:  # time_lst[i] < pairs[j][0]
            if remove_lights:
                inds.append(i)
            i += 1

    return inds

def combine_list(pairs, margin=10):
    """

    Parameters
    ----------
    pairs : Iterable of pairs representing times of stimuli
    margin : Time in ms to add before and after every stimuli

    Returns
    -------
    np.ndarray of the light stimuli with no intersection
    """
    if len(pairs) == 0:
        return np.array([[-2, -1]])
    pairs = pairs + [-margin, margin]
    inds = np.argsort(pairs[:, 0])
    ret = []
    min_p, max_p = pairs[inds][0]
    for pair in pairs[inds][1:]:
        if min_p <= pair[0] <= max_p:
            max_p = max(pair[1], max_p)
        else:
            ret.append([min_p, max_p])
            min_p, max_p = pair
    ret.append([min_p, max_p])
    return np.array(ret)

def remove_light(cluster, remove_lights, data_path):
    """

    Parameters
    ----------
    cluster : Cluster to update
    remove_lights : Bool; remove spikes during stimuli or remove spikes not during stimuli
    data_path : String; path to raw data

    Returns
    -------
    Iterable of integers; valid indices
    """
    pairs = io.loadmat(data_path + f'/{cluster.filename}/{cluster.filename}.stm', simplify_cells=True)['pairs']
    pairs = combine_list(pairs)

    timings = cluster.timings
    inds = get_idnds(timings, pairs, remove_lights)

    return inds
