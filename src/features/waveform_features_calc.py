import numpy as np
from data_utils.clusters import Spike
import time
from constants import VERBOS
from features.spatial_features_calc import wavelet_transform, calc_amps

from features.waveform_features.FET_break import BreakMeasurement
from features.waveform_features.FET_fwhm import FWHM
from features.waveform_features.FET_get_acc import GetAcc
from features.waveform_features.FET_max_speed import MaxSpeed
from features.waveform_features.FET_peak2peak import Peak2Peak
from features.waveform_features.FET_rise_coef import RiseCoef
from features.waveform_features.FET_smile_cry import SmileCry

features = [BreakMeasurement(), FWHM(), GetAcc(), MaxSpeed(), Peak2Peak(), RiseCoef(), SmileCry()]


def get_main_channels(chunks):
    """

    Parameters
    ----------
    chunks : Mean waveform for each chunk over all 8 channels

    Returns
    -------
    Iterable of Spike only for the main channels
    """
    ret = []

    for i, chunk in enumerate(chunks):
        chunk_data = chunk.get_data()
        chunk_amp = chunk_data.max(axis=1) - chunk_data.min(axis=1)
        main_channel = np.argmax(chunk_amp)
        data = chunk_data[main_channel].copy() / abs(chunk_data[main_channel].min())  # scale
        ret.append(Spike(data=data))  # set main channel to be the one with highest peak - trough

    return ret


def calc_wf_features(chunks, transform=None):
    """

    Parameters
    ----------
    chunks : List of chunks (each chunk is an 8-channel spike)
    transform : Bool; whether to perform event-based delta transformation

    Returns
    -------
    A matrix of waveform-based features for the chunks
    """
    feature_mat_for_cluster = None

    if transform is not None:
        chunks = wavelet_transform(chunks, transform, calc_amps(chunks))

    main_chunks = get_main_channels(chunks)

    for feature in features:
        start_time = time.time()
        mat_result = feature.calculate_feature(main_chunks)
        if feature_mat_for_cluster is None:
            feature_mat_for_cluster = mat_result
        else:
            feature_mat_for_cluster = np.concatenate((feature_mat_for_cluster, mat_result), axis=1)
        end_time = time.time()
        if VERBOS:
            print(f"feature {feature.name} processing took {end_time - start_time:.3f} seconds")

    return feature_mat_for_cluster


def get_wf_features_names():
    """

    Returns
    -------
    List of the names of the waveform-based features
    """
    names = []
    for feature in features:
        names += feature.headers
    return names
