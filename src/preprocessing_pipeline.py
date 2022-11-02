import time
import warnings
from os import listdir, mkdir
from os.path import isdir, isfile, join, exists

import numpy as np
import pandas as pd
import scipy.io as io
from tqdm import tqdm

from constants import UPSAMPLE, VERBOS, SEED, CHUNK_SIZES, TRANS_WF, RICH_FACTOR
from data_utils.clusters import Spike, Cluster
from data_utils.upsampling import upsample_spike
from features.spatial_features_calc import DELTA_MODE
# import the different features
from features.spatial_features_calc import calc_spatial_features, get_spatial_features_names
from features.spike_timing_features_calc import calc_st_features, get_st_features_names
from features.waveform_features_calc import calc_wf_features, get_wf_features_names
from paths import DATA_TEMP_PATH, SAVE_PATH, DATA_MAT_PATH, SAVE_PATH_RICH

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

def load_clusters(load_path):
    """
    Parameters
    ----------
    load_path : string; path to load clusters from

    Returns
    -------
    yields wrapped clusters
    """
    files_list = [DATA_TEMP_PATH + f for f in listdir(load_path) if isfile(join(load_path, f))]
    clusters = set()
    for file in tqdm(files_list):
        start_time = time.time()
        path_elements = file.split('/')[-1].split('__')
        if 'timing' in path_elements[-1]:
            continue
        unique_name = path_elements[0]
        if unique_name not in clusters:
            clusters.add(unique_name)
        else:
            raise Exception('Duplicate file in load path')
        cluster = Cluster()
        cluster.load_cluster(file)
        timing_file = file.replace('spikes', 'timings')
        cluster.load_cluster(timing_file)

        assert cluster.assert_legal()

        end_time = time.time()

        if len(cluster.np_spikes) == 0:
            continue

        if VERBOS:
            print(f"cluster loading took {end_time - start_time:.3f} seconds")
        yield [cluster]


def create_chunks(cluster, spikes_in_waveform, seed):
    """

    Parameters
    ----------
    cluster : cluster to break into chunks
    spikes_in_waveform : iterable of integers; chunk sizes to use
    seed : seed for random partitioning

    Returns
    -------
    Iterable of Spike; indices used in each spike
    """
    ret_spikes = []
    ret_inds = []
    # for each chunk size create the data
    for chunk_size in spikes_in_waveform:
        if chunk_size == 0:  # unit based approach
            mean = cluster.calc_mean_waveform()
            ret_spikes.append([mean])
            ret_inds.append(np.expand_dims(np.arange(len(cluster.spikes)), axis=0))
        elif chunk_size == 1:  # chunk based approach with raw spikes
            ret_spikes.append(cluster.spikes)
            ret_inds.append(np.expand_dims(np.arange(len(cluster.spikes)), 1))
        else:  # chunk based approach
            if cluster.np_spikes is None:  # this is done for faster processing
                cluster.finalize_cluster()
            spikes = cluster.np_spikes
            inds = np.arange(spikes.shape[0])
            np.random.seed(seed)
            np.random.shuffle(inds)
            spikes = spikes[inds]
            k = spikes.shape[0] // chunk_size  # number of chunks
            if k == 0:  # cluster size is larger than the number of spikes in this cluster, same as chunk size of 0
                ret_spikes.append([cluster.calc_mean_waveform()])
                ret_inds.append(np.array([np.arange(len(cluster.spikes))]))
                continue
            chunks = np.array_split(spikes, k)  # split the data into k chunks of minimal size of chunk_size

            ret_inds.append(np.array(np.array_split(inds, k)))
            res = []
            for chunk in chunks:
                res.append(Spike(data=chunk.mean(axis=0)))  # take the average spike
            ret_spikes.append(res)

    return ret_spikes, ret_inds


def get_clu_reg(cluster, mat_file):
    """

    Parameters
    ----------
    cluster : The cluster
    mat_file : String; path to mat file containing tagging information

    Returns
    -------
    The region code of the cluster
    """
    mat = io.loadmat(mat_file, simplify_cells=True)
    shank, clu = cluster.shank, cluster.num_within_file
    filename = cluster.filename
    for i in range(len(mat['filename'])):
        if mat['filename'][i] != filename:
            continue
        else:
            if shank == mat['shankclu'][i][0] and clu == mat['shankclu'][i][1]:
                return mat['region'][i]
    raise KeyError


def main(chunk_sizes, save_path, mat_file, load_path, trans_wf, seed):
    """

    Parameters
    ----------
    chunk_sizes : Iterable of integers; chunk sizes to extract
    save_path : String; path to save csvs of extracted features
    mat_file : String; path to the tagging information mat
    load_path : String; path to read the clusters from
    trans_wf : Bool; perform transformation and extract only waveform-based features or perform regular pipeline
    seed : Int; seed to determine randomized chunking process

    Returns
    -------
    None. Outputs are saved as csv files
    """
    clusters_generator = load_clusters(load_path)

    # define headers for saving later
    if not trans_wf:
        headers = get_spatial_features_names()
        headers += get_wf_features_names()
        headers += get_st_features_names()
    else:
        original_names = get_wf_features_names()
        headers = []
        for inst in DELTA_MODE:
            headers += [f"{inst.name}_{name}" for name in original_names]
    headers += ['max_abs', 'name', 'region', 'label']

    for clusters in clusters_generator:
        for cluster in clusters:  # for each unit
            if cluster.label < 0:
                print('Skipping cluster:' + cluster.get_unique_name() + ' for not being PYR nor PV')
                continue
            print('Processing cluster:' + cluster.get_unique_name())

            max_abs = np.absolute(cluster.calc_mean_waveform().get_data()).max()
            region = get_clu_reg(cluster, mat_file)
            # print('Dividing data to chunks...')
            start_time = time.time()
            spike_chunks, ind_chunks = create_chunks(cluster, chunk_sizes, seed)
            end_time = time.time()
            if VERBOS:
                print(f"chunk creation took {end_time - start_time:.3f} seconds")

            path = save_path + 'mean_spikes'
            if not isdir(path):
                mkdir(path)
            path += '/' + cluster.get_unique_name()
            np.save(path, cluster.calc_mean_waveform().data)

            for chunk_size, rel_data, inds in zip(chunk_sizes, spike_chunks, ind_chunks):
                path = save_path + str(chunk_size)
                if exists(path + '/' + cluster.get_unique_name() + ".csv"):
                    continue
                # upsample
                rel_data = [Spike(data=upsample_spike(spike.data, UPSAMPLE))
                            for spike in rel_data]
                if not trans_wf:
                    st_features_mat = calc_st_features(cluster.timings, inds)
                    spatial_features_mat = calc_spatial_features(rel_data)
                    wf_features_mat = calc_wf_features(rel_data)
                    feature_mat_for_cluster = np.concatenate((spatial_features_mat, wf_features_mat, st_features_mat),
                                                             axis=1)
                else:
                    feature_mat_for_cluster = None
                    for inst in DELTA_MODE:
                        temp_features_mat = calc_wf_features(rel_data, inst)
                        if feature_mat_for_cluster is None:
                            feature_mat_for_cluster = temp_features_mat
                        else:
                            feature_mat_for_cluster = np.concatenate((feature_mat_for_cluster, temp_features_mat),
                                                                     axis=1)

                # Append metadata for the cluster
                max_abss = np.ones((len(rel_data), 1)) * max_abs
                feature_mat_for_cluster = np.concatenate((feature_mat_for_cluster, max_abss), axis=1)

                names = np.ones((len(rel_data), 1), dtype=object) * cluster.get_unique_name()
                feature_mat_for_cluster = np.concatenate((feature_mat_for_cluster, names), axis=1)

                regions = np.ones((len(rel_data), 1)) * region
                feature_mat_for_cluster = np.concatenate((feature_mat_for_cluster, regions), axis=1)

                labels = np.ones((len(rel_data), 1)) * cluster.label
                feature_mat_for_cluster = np.concatenate((feature_mat_for_cluster, labels), axis=1)

                # Save the data to a separate file (one for each cluster)
                if not isdir(path):
                    mkdir(path)
                path += '/' + cluster.get_unique_name() + ".csv"
                df = pd.DataFrame(data=feature_mat_for_cluster)
                df.to_csv(path_or_buf=path, index=False, header=headers)  # save to csv
            print('saved clusters to csv')

def add_chunk_statistics():
    """
    Adds chunk statistics to the calculated features
    Returns
    -------
    None. output is saved as csv files in SAVE_PATH_RICH
    """
    input_list = [f'{SAVE_PATH}{cs}' for cs in CHUNK_SIZES]
    output_list = [f'{SAVE_PATH_RICH}{cs}' for cs in CHUNK_SIZES]
    if not isdir(SAVE_PATH_RICH):
        mkdir(SAVE_PATH_RICH)
    for d in output_list:
        if not isdir(d):
            mkdir(d)

    drop = ['num_spikes', 'max_abs', 'name', 'region', 'label']

    for p, d in zip(input_list, output_list):
        files = listdir(p)
        for file in sorted(files):
            df = pd.read_csv(p + '/' + file)

            n = df.shape[0]
            md = df[drop]
            df = df.drop(columns=drop)
            nc = len(df.columns)

            avg = np.expand_dims(df.mean().to_numpy(), axis=0).repeat(n, axis=0)  # 1
            std = np.expand_dims(df.std().to_numpy(), axis=0).repeat(n, axis=0)  # 2
            q25 = np.expand_dims(df.quantile(0.25).to_numpy(), axis=0).repeat(n, axis=0)  # 3
            q50 = np.expand_dims(df.quantile(0.5).to_numpy(), axis=0).repeat(n, axis=0)  # 4
            q75 = np.expand_dims(df.quantile(0.75).to_numpy(), axis=0).repeat(n, axis=0)  # 5

            new_headers = []
            for c in df.columns:
                new_headers += [f'{c}', f'{c}_avg', f'{c}_std', f'{c}_q25', f'{c}_q50', f'{c}_q75']
            new_headers += list(md.columns)

            new_df = np.zeros((n, nc * RICH_FACTOR + len(drop)), object)
            new_df[:, :-5:RICH_FACTOR] = df.to_numpy()
            new_df[:, 1:-5:RICH_FACTOR] = avg
            new_df[:, 2:-5:RICH_FACTOR] = std
            new_df[:, 3:-5:RICH_FACTOR] = q25
            new_df[:, 4:-5:RICH_FACTOR] = q50
            new_df[:, 5:-5:RICH_FACTOR] = q75
            new_df[:, -5:] = md.to_numpy()

            new_df = pd.DataFrame(new_df, columns=new_headers)

            new_df.to_csv(path_or_buf=d + '/' + file, index=False, header=new_headers)


if __name__ == "__main__":
    if not isdir(SAVE_PATH):
        mkdir(SAVE_PATH)

    main(CHUNK_SIZES, SAVE_PATH, DATA_MAT_PATH, DATA_TEMP_PATH, TRANS_WF, SEED)

    if not TRANS_WF:
        add_chunk_statistics()
