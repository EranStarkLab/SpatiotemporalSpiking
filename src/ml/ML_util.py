import os
import math
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from constants import SEED


def split_features(data):
    """
   The function separates the features and the labels of the clusters
   """
    return data[:, :-1], data[:, -1].astype('int32')


def is_legal(cluster):
    """
   This function determines whether or not a cluster's label is legal (PYR/PV). It is assumed that all waveforms
   of the cluster have the same label.
   """
    row = cluster[0]
    return row[-1] >= 0


def read_data(path, keep=None, filter_inp=None):
    """
    The function reads the data from all files in the path.
    It is assumed that each file represents a single cluster, and have some number of waveforms.
    Parameters
    ----------
    path : String; path to raw data
    keep : indices of features to use
    filter_inp : optional; set of unit indices to use

    Returns
    -------

    """
    if keep is None:
        keep = []
    files = os.listdir(path)
    clusters = []
    names = []
    regions = []
    filter_set = set() if filter_inp is None else filter_inp
    for i, file in enumerate(sorted(files)):
        df = pd.read_csv(path + '/' + file)
        name = df.name[0]
        region = df.region[0]
        df = df.drop(columns=['name', 'region'])
        nd = df.to_numpy(dtype='float64')

        if filter_inp is not None:
            if i in filter_inp:
                if len(keep) > 0:  # i.e. keep != []
                    nd = nd[:, keep]
                clusters.append(nd)
            else:
                continue
        elif is_legal(nd):
            if len(keep) > 0:  # i.e. keep != []
                nd = nd[:, keep]
            clusters.append(nd)
            filter_set.add(i)
        else:
            continue

        names.append(name)
        regions.append(region)
    return np.asarray(clusters), np.array(names), np.array(regions), filter_set


def break_data(data, cluster_names):
    """
   The function receives unordered data and returns a list with three numpy arrays: 1) with all the pyramidal clusters,
   2) with all the PV clusters and 3) with all the unlabeled clusters
   """
    pyr_inds = get_inds(data, 1)
    in_inds = get_inds(data, 0)
    ut_inds = get_inds(data, -1)
    ret = [data[pyr_inds], data[in_inds], data[ut_inds]]
    names = [cluster_names[pyr_inds], cluster_names[in_inds], cluster_names[ut_inds]]
    return ret, names


def was_created(paths, per_train, per_test2, per_test1):
    """
   The function checks if all datasets were already created and return True iff so
   """
    for path in paths:
        path = path + str(per_train) + str(per_test2) + str(per_test1)
        if not os.path.isdir(path):
            return False
    return True


def create_datasets(per_train, per_test2, per_test1, raw_path, chunks, save_path, keep,
                    verbos=False, seed=None, region_based=False, train_ca1=True):
    """

    Parameters
    ----------
    per_train : Fraction of samples in train set
    per_test2 : Fraction of samples in test set 2 (not relevant if region_based is True)
    per_test1 : Fraction of samples in test set 1
    raw_path : String; path to post-processed data
    chunks : chunk sizes
    save_path : String; path where datasets should be saved
    keep : indices of features to keep
    verbos : Bool; indicating if should print extra information
    seed : seed for random partitioning
    region_based : Bool; indicating if should perform region-based partitioning (see paper)
    train_ca1 : Bool; relevant only if region_based is True, then determines the training region

    Returns
    -------
    None. Datasets are saved
    """
    if keep is None:
        keep = []

    names = [f"{chunk_size}" + '_' for chunk_size in chunks]
    paths = [raw_path + f'{chunks_size}' for chunks_size in chunks]

    should_load = was_created([save_path + '/' + name for name in names], per_train, per_test2, per_test1)

    inds = []
    inds_initialized = False
    filter_set = None
    for name, path in zip(names, paths):
        if not should_load:
            print('Reading data from %s...' % path)
            data, cluster_names, regions, filter_set = read_data(path, keep=keep, filter_inp=filter_set)
            if not region_based:
                data, cluster_names = break_data(data, cluster_names)
                if not inds_initialized:
                    if seed is None:
                        np.random.seed(SEED)
                    else:
                        np.random.seed(seed)
                    for c in data:
                        inds_temp = np.arange(c.shape[0])
                        np.random.shuffle(inds_temp)
                        inds.append(inds_temp)
                    inds_initialized = True

                data = [c[inds[i]] for i, c in enumerate(data)]
                cluster_names = [c[inds[i]] for i, c in enumerate(cluster_names)]
        else:
            data = None  # only because we need to send something to split_data()
            cluster_names = None
            regions = None
        print('Splitting %s set...' % name)
        split_data(data, cluster_names,  per_train=per_train, per_test2=per_test2, per_test1=per_test1,
                   path=save_path, data_name=name, should_load=should_load, verbos=verbos, seed=seed,
                   region_based=region_based, regions=regions, train_ca1=train_ca1)


def get_dataset(path, verbose=False):
    """
   This function simply loads a dataset from the path and returns it. It is assumed that create_datasets() was already
   executed
   """
    if verbose:
        print('Loading data set from %s...' % path)
    train = np.load(path + 'train.npy', allow_pickle=True)
    test2 = np.load(path + 'test2.npy', allow_pickle=True)
    test1 = np.load(path + 'test1.npy', allow_pickle=True)
    train_names = np.load(path + 'train_names.npy', allow_pickle=True)
    test2_names = np.load(path + 'test2_names.npy', allow_pickle=True)
    test1_names = np.load(path + 'test1_names.npy', allow_pickle=True)

    data = np.concatenate((train, test2, test1))

    num_clusters = data.shape[0]
    num_wfs = count_waveforms(data)
    if verbose:
        print_data_stats(train, 'train', num_clusters, num_wfs)
        print_data_stats(test2, 'test2', num_clusters, num_wfs)
        print_data_stats(test1, 'test1', num_clusters, num_wfs)

    return train, test2, test1, train_names, test2_names, test1_names


def take_partial_data(data, start, end):
    """
   The function receives data which is a list with three numpy arrays: 1) clusters with pyramidal label, 2) clusters
   with PV label and 3) unlabeled clusters. It returns a numpy array of clusters consisting of all parts of the
   data made from the start to end percentiles of the original elements of the data.
   """
    len0 = len(data[0])
    len1 = len(data[1])
    len2 = len(data[2])
    ret = np.concatenate((data[0][math.floor(start * len0): math.floor(end * len0)],
                          data[1][math.floor(start * len1): math.floor(end * len1)],
                          data[2][math.floor(start * len2): math.floor(end * len2)]))
    return ret


def take_region_data(data, names, region, regions):
    if region:
        inds = np.argwhere(regions >= 1).flatten()
    else:
        inds = np.argwhere(regions == 0).flatten()
    return data[inds], names[inds]


def split_region_data(data, names, seed, test_per):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_per, random_state=seed)
    n_samples = len(data)
    labels = np.array([row[0][-1] for row in data])
    train_inds, dev_inds = next(sss.split(np.zeros(n_samples), labels))
    return data[train_inds], names[train_inds], data[dev_inds], names[dev_inds]


def split_data(data, names, per_train=0.6, per_test2=0.2, per_test1=0.2, path='../data_sets', should_load=True,
               data_name='', verbos=False, seed=None, region_based=False, regions=None, train_ca1=True):
    """
    The function splits the data to non-overlapping sets
    Parameters
    ----------
    data : Feature information
    names : Names of recording sessions associated with the samples
    per_train : Fraction of samples in train set
    per_test2 : Fraction of samples in test set 2 (not relevant if region_based is True)
    per_test1 : Fraction of samples in test set 1
    path : String; path where datasets should be saved
    should_load : Bool; load instead of split if possible
    data_name : STRING; identifier of the dataset for saving
    verbos : Bool; indicating if should print extra information
    seed : seed for random partitioning
    region_based : Bool; indicating if should perform region-based partitioning (see paper)
    regions : Regions of units recorded associated with the samples
    train_ca1 : Bool; relevant only if region_based is True, then determines the training region

    Returns
    -------
    The three sets created are saved and returned
    """
    assert per_train + per_test2 + per_test1 == 1
    name = data_name + str(per_train) + str(per_test2) + str(per_test1) + '/'
    full_path = path + '/' + name if path is not None else None
    if path is not None and os.path.exists(full_path) and should_load:
        print('Loading data set from %s...' % full_path)
        train = np.load(full_path + 'train.npy', allow_pickle=True)
        test2 = np.load(full_path + 'test2.npy', allow_pickle=True)
        test1 = np.load(full_path + 'test1.npy', allow_pickle=True)
    else:
        per_test2 += per_train

        if region_based:
            train, train_names = take_region_data(data, names, True and train_ca1, regions)
            train, train_names, test2, test2_names = split_region_data(train, train_names, seed, per_test1)
            test1, test1_names = take_region_data(data, names, not train_ca1, regions)
        else:
            train = take_partial_data(data, 0, per_train)
            train_names = take_partial_data(names, 0, per_train)
            test2 = take_partial_data(data, per_train, per_test2)
            test2_names = take_partial_data(names, per_train, per_test2)
            test1 = take_partial_data(data, per_test2, 1)
            test1_names = take_partial_data(names, per_test2, 1)

        if path is not None:
            try:
                if not os.path.exists(full_path):
                    os.mkdir(full_path)
            except OSError:
                print("Creation of the directory %s failed, not saving set" % full_path)
            else:
                print("Successfully created the directory %s now saving data set" % full_path)
                np.save(full_path + 'train', train)
                np.save(full_path + 'test2', test2)
                np.save(full_path + 'test1', test1)
                np.save(full_path + 'train_names', train_names)
                np.save(full_path + 'test2_names', test2_names)
                np.save(full_path + 'test1_names', test1_names)

    if verbos:
        data = np.concatenate((train, test2, test1))
        num_clusters = data.shape[0]
        num_wfs = count_waveforms(data)
        print_data_stats(train, 'train', num_clusters, num_wfs)
        print_data_stats(test2, 'test2', num_clusters, num_wfs)
        print_data_stats(test1, 'test1', num_clusters, num_wfs)

    return train, test2, test1


def print_data_stats(data, name, total_clusters, total_waveforms):
    """
   This function prints various statistics about the given set
   """
    if len(data) == 0:
        print('No examples in %s set' % name)
        return
    num_clstr = data.shape[0]
    num_wfs = count_waveforms(data)
    clstr_ratio = num_clstr / total_clusters
    wfs_ratio = num_wfs / total_waveforms
    print('Total number of clusters in %s data is %d (%.3f%%) consisting of %d waveforms (%.3f%%)'
          % (name, num_clstr, 100 * clstr_ratio, num_wfs, 100 * wfs_ratio))

    pyr_clstrs = data[get_inds(data, 1)]
    num_pyr_clstr = pyr_clstrs.shape[0]
    ratio_pyr_clstr = num_pyr_clstr / num_clstr
    num_pyr_wfs = count_waveforms(pyr_clstrs)
    pyr_wfs_ratio = num_pyr_wfs / num_wfs
    print('Total number of pyramidal clusters in %s data is %d (%.3f%%) consisting of %d waveforms (%.3f%%)'
          % (name, num_pyr_clstr, 100 * ratio_pyr_clstr, num_pyr_wfs, 100 * pyr_wfs_ratio))

    in_clstrs = data[get_inds(data, 0)]
    num_in_clstr = in_clstrs.shape[0]
    ratio_in_clstr = num_in_clstr / num_clstr
    num_in_wfs = count_waveforms(in_clstrs)
    in_wfs_ratio = num_in_wfs / num_wfs
    print('Total number of interneurons clusters in %s data is %d (%.3f%%) consisting of %d waveforms (%.3f%%)'
          % (name, num_in_clstr, 100 * ratio_in_clstr, num_in_wfs, 100 * in_wfs_ratio))

    ut_clstrs = data[get_inds(data, -1)]
    num_ut_clstr = ut_clstrs.shape[0]
    ratio_ut_clstr = num_ut_clstr / num_clstr
    num_ut_wfs = count_waveforms(ut_clstrs)
    ut_wfs_ratio = num_ut_wfs / num_wfs
    print('Total number of untagged clusters in %s data is %d (%.3f%%) consisting of %d waveforms (%.3f%%)'
          % (name, num_ut_clstr, 100 * ratio_ut_clstr, num_ut_wfs, 100 * ut_wfs_ratio))


def get_inds(data, label):
    """
   The function recieved a numpy array of clusters (numpy arrays of varying sizes) and returns
   the indeces with the given label. If the label is -1 all clusters with negative (i.e. untagged)
   labels indices are returned.
   This function is needed as numpy has hard time working with varying size arrays.
   """
    inds = []
    for ind, cluster in enumerate(data):
        if label >= 0:
            if cluster[0, -1] == label:
                inds.append(ind)
        else:
            if cluster[0, -1] < 0:
                inds.append(ind)
    return inds


def count_waveforms(data):
    """
   This function counts the number of waveforms in all clusters of the data.
   The main usage of this function is statistical data gathering.
   """
    counter = 0
    for cluster in data:
        counter += cluster.shape[0]
    return counter

def squeeze_clusters(data):
    """
   This function receives an nd array with elements with varying sizes.
   It removes the first dimension.
   As numpy doesn't nicely support varying sizes we implement what otherwise could have been achieved using reshape or
   squeeze
   """
    res = []
    for cluster in data:
        for waveform in cluster:
            res.append(waveform)
    return np.asarray(res)
