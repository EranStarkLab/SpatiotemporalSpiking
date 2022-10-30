import pandas as pd
import numpy as np

from constants import SPATIAL_R, feature_names_rich
from constants import SPATIAL, WAVEFORM, SPIKE_TIMING
from constants import RICH_FACTOR, MODALITIES, CHUNK_SIZES, NUM_EVENTS

# Note that the warning: "RuntimeWarning: Mean of empty slice." is expected for multiple modalities

NUM_FETS = len(SPATIAL[:-1]) + len(WAVEFORM[:-1]) + len(SPIKE_TIMING[:-1])
NUM_MOMENTS = RICH_FACTOR + 1
NUM_MODALITIES = len(MODALITIES)
NUM_CHUNKS = len(CHUNK_SIZES)
EVENTS = ['FMC', 'NEG', 'SMC']


def get_group_imp(inds, arr, full=False):
    if full:  # makes it so that we can sum easily over the entire row without getting NaNs
        arr = np.nan_to_num(arr)
    arr_m = abs(arr[:, :, inds].sum(axis=2))
    fam_imps = np.asarray([a[~np.isnan(a)].mean() for a in arr_m])
    return fam_imps

def get_feature_imp(df, imps):
    for i in range(NUM_FETS):
        inds = [i * (NUM_MOMENTS + 1) + j for j in range(NUM_MOMENTS + 1)]
        new_imp = get_group_imp(inds, imps)
        df.loc[:, f'test feature new {i + 1}'] = new_imp

    drop = [f'test feature {i + 1}' for i in range(NUM_FETS * (NUM_MOMENTS + 1))] +\
           [f'test2 feature {i + 1}' for i in range(NUM_FETS * (NUM_MOMENTS + 1))]
    df = df.drop(columns=drop)
    mapper = {f'test feature new {i + 1}': f'test feature {i + 1}' for i in range(NUM_FETS)}
    df = df.rename(columns=mapper)
    return df


def get_stat_imp(df, imps):
    for i in range(NUM_MOMENTS + 1):
        inds = [j * (NUM_MOMENTS + 1) + i for j in range(NUM_FETS)]
        new_imp = get_group_imp(inds, imps, full=True)
        df.loc[:, f'test feature new {i + 1}'] = new_imp

    drop = [f'test feature {i + 1}' for i in range(NUM_FETS * (NUM_MOMENTS + 1))] +\
           [f'test2 feature {i + 1}' for i in range(NUM_FETS * (NUM_MOMENTS + 1))]
    df = df.drop(columns=drop)
    mapper = {f'test feature new {i + 1}': f'test feature {i + 1}' for i in range(NUM_MOMENTS + 1)}
    df = df.rename(columns=mapper)
    return df


def get_events_imp(df, imps):
    spatial_inds = np.arange(imps.shape[0]).reshape((imps.shape[0] // NUM_CHUNKS, NUM_CHUNKS))[
                   ::NUM_MODALITIES].flatten()
    imps = imps[spatial_inds]
    df = df[df.modality == 'spatial']

    spatial_fet_names = [feature_names_rich[i] for i in SPATIAL_R[:-1]]

    for i, event in enumerate(EVENTS):
        inds = [spatial_fet_names.index(name) for name in spatial_fet_names if event in name]
        new_imp = get_group_imp(inds, imps)
        df.loc[:, f'test feature new {i + 1}'] = new_imp

    drop = [f'test feature {i + 1}' for i in range(NUM_FETS * (NUM_MOMENTS + 1))] + [f'test2 feature {i + 1}' for i in
                                                                                     range(
                                                                                         NUM_FETS * (NUM_MOMENTS + 1))]
    df = df.drop(columns=drop)
    mapper = {f'test feature new {i + 1}': f'test feature {i + 1}' for i in range(NUM_EVENTS)}
    df = df.rename(columns=mapper)
    return df

def get_family_imp(df, imps):
    spatial_inds = np.arange(imps.shape[0]).reshape((imps.shape[0] // NUM_CHUNKS, NUM_CHUNKS))[
                   ::NUM_MODALITIES].flatten()
    imps = imps[spatial_inds]
    df = df[df.modality == 'spatial']

    spatial_families = get_spatial_families_dict()

    spatial_fet_names = [feature_names_rich[i] for i in SPATIAL_R[:-1]]

    for i, fam in enumerate(spatial_families):
        inds = [spatial_fet_names.index(name) for name in spatial_families[fam]]
        new_imp = get_family_imp(inds, imps)
        df_c[f'test feature new {i + 1}'] = new_imp

    drop = [f'test feature {i + 1}' for i in range(NUM_FETS * (NUM_MOMENTS + 1))] + [f'test2 feature {i + 1}' for i in
                                                                                     range(
                                                                                         NUM_FETS * (NUM_MOMENTS + 1))]
    df = df.drop(columns=drop)
    mapper = {f'test feature new {i + 1}': f'test feature {i + 1}' for i in range(len(spatial_families))}
    df = df.rename(columns=mapper)

    return df


def get_spatial_families_dict():
    spatial_families_temp = {'value-based': ["SPD_Count", "SPD_SD", "SPD_Area"],
                             'time-based': ["NEG_Time-lag_SS", "NEG_Time-lag_SD", "FMC_Time-lag_SS","FMC_Time-lag_SD",
                                            "SMC_Time-lag_SS", "SMC_Time-lag_SD"],
                             'graph-based': ["NEG_Graph_Average_weight", "NEG_Graph_Shortest_path",
                                             "NEG_Graph_Longest_path", "FMC_Graph_Average_weight",
                                             "FMC_Graph_Shortest_path", "FMC_Graph_Longest_path",
                                             "SMC_Graph_Average_weight", "SMC_Graph_Shortest_path",
                                             "SMC_Graph_Longest_path"]}

    spatial_families = dict()
    for key in spatial_families_temp:
        temp_list = []
        for f in spatial_families_temp[key]:
            temp_list += [f'{f}', f'{f}_avg', f'{f}_std', f'{f}_q25', f'{f}_q50', f'{f}_q75']
        spatial_families[key] = temp_list

    return spatial_families

def combine_chunks(df_c, df_0, imps_c):
    df_c = get_feature_imp(df_c, imps_c)
    df_0 = df_0[df_0.chunk_size == 0]

    df_c[df_c.chunk_size == 0] = df_0
    return df_c
