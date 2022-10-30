import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import numpy as np
import pandas as pd
import scipy.signal as signal
import scipy.stats as stats
import scipy.io as io
from sklearn.preprocessing import StandardScaler
from os import listdir, mkdir
from os.path import isfile, isdir, join
import seaborn as sns

from data_utils.clusters import Cluster, Spike
from features.spike_timing_features_calc import calc_temporal_histogram
from features.spatial_features_calc import DELTA_MODE
from features.spatial_features_calc import calc_pos as calc_pos_imp
from features.spatial_features.FET_depolarization_graph import calculate_distances_matrix
from data_utils.upsampling import upsample_spike
from vis_utils.plot_tests import plot_fet_imp, plot_cf_thr, plot_roc_curve, plot_test_vs_test2_bp, plot_auc_chunks_bp
from vis_utils.calc_psth import main as disp_psth
from vis_utils.calc_cch import main as disp_cch

from constants import PV_COLOR, LIGHT_PV, PYR_COLOR, LIGHT_PYR
from constants import SPATIAL, WAVEFORM, SPIKE_TIMING, feature_names_org
from constants import CHUNK_SIZES, MODALITIES
from constants import COORDINATES, UPSAMPLE, TIMESTEPS, NUM_CHANNELS, NUM_EVENTS
from constants import PYR_NAME, PV_NAME
from constants import BEST_WF_CHUNK, BEST_ST_CHUNK, BEST_SPATIAL_CHUNK
from constants import VOL_RANGE, AMPLIFICATION, NBITS, SAMPLE_RATE

from paths import SAVE_PATH, SAVE_PATH_WF_TRANS, DATA_PATH, STATS_PATH, DATA_TEMP_PATH, TEMP_PATH_FULL, FIG_PATH
from paths import MAIN_RES, MAIN_PREDS, BASE_RES, EVENTS_RES, BASE_EVENTS_RES, TRANS_WF_RES, TRANS_WF_PREDS
from paths import REGION_CA1_RES, REGION_NCX_RES, CORR_MAT, MI_MAT

import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

feature_mapping = {feature_names_org[i]: i for i in range(len(feature_names_org))}


def clear():
    plt.clf()
    plt.cla()
    plt.close('all')


def load_cluster(load_path, name):
    files = [load_path + f for f in listdir(load_path) if isfile(join(load_path, f)) and name + '_' in f]
    spikes_f, timimg_f = files
    if 'timing' not in timimg_f:
        spikes_f, timimg_f = timimg_f, spikes_f
    cluster = Cluster()
    cluster.load_cluster(spikes_f)
    cluster.load_cluster(timimg_f)
    assert cluster.assert_legal()
    return cluster


def pie_chart(df):
    labels = ['PYR CA1', 'PYR nCX', 'PV nCX', 'PV CA1']
    colors = [PYR_COLOR, LIGHT_PYR, LIGHT_PV, PV_COLOR]

    df.region = df.region * 2
    df['Label X Region'] = df.label + df.region

    x = df['Label X Region'].to_numpy()

    sizes = [np.count_nonzero(x == 3), np.count_nonzero(x == 1), np.count_nonzero(x == 0), np.count_nonzero(x == 2)]
    total = sum(sizes)

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct=lambda p: '{:.0f}'.format(p * total / 100), colors=colors, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.savefig(FIG_PATH + f"Fig1_C_pie.pdf", transparent=True)
    clear()


def trans_units(cluster):
    data = cluster.spikes
    data = (data * VOL_RANGE * 1e6) / (AMPLIFICATION * (2 ** NBITS))  # 1e6 for micro V
    cluster.spikes = data
    cluster.np_spikes = data


def get_main(chunks):
    chunk_amp = chunks.max(axis=1) - chunks.min(axis=1)
    main_channel = np.argmax(chunk_amp)
    return main_channel


def TTP_dur(clu, color, name):
    chunks = clu.calc_mean_waveform()
    if UPSAMPLE != 1:
        chunks = upsample_spike(chunks.data)

    spike = chunks[get_main(chunks)]
    spike = spike / abs(spike.min())

    dep_ind = spike.argmin()
    dep = spike[dep_ind]
    hyp_ind = spike.argmax()
    hyp = spike[hyp_ind]
    t2p = 1.6 * (hyp_ind - dep_ind) / len(spike)
    print(f"{name} TTP-duration is {t2p: .3g} ms")  # 1.6 ms is the time of every spike (32 ts in 20khz)

    fig, ax = plt.subplots()
    ax.plot(spike, c=color)
    ax.plot([dep_ind, hyp_ind], [dep, hyp], marker='o', linestyle='None', c=color)

    plt.savefig(FIG_PATH + f"Fig2_Aa_{name}_TTP_duration.pdf", transparent=True)
    clear()

    return t2p


def load_df(features=None, trans_labels=True, path=SAVE_PATH + '0/'):
    df = None
    files = listdir(path)
    for file in sorted(files):
        if df is None:
            df = pd.read_csv(path + '/' + file)
        else:
            temp = pd.read_csv(path + '/' + file)
            df = df.append(temp)

    df = df.loc[df.label >= 0]
    df.region = df.region.map({0: 0, 1: 1, 3: 1, 4: 1})
    if trans_labels:
        df.label = df.label.map({1: 'PYR', 0: 'PV'})
    if features is not None:
        df = df[features]

    return df


def disp_corr_mat(df, name, modality, order):
    pvals = io.loadmat(CORR_MAT)['pvals']
    inds = [feature_mapping[fet_name] for fet_name in order]
    pvals = pvals[inds][:, inds]

    annotations = np.where(pvals < 0.05, '*', '')
    annotations = np.where(pvals < 0.01, '**', annotations)
    annotations = np.where(pvals < 0.001, '***', annotations)

    correlation_matrix = df[order].corr(method="spearman")
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    plt.yticks(rotation=30)
    cmap = sns.color_palette("vlag", as_cmap=True)
    _ = sns.heatmap(correlation_matrix, annot=annotations, mask=mask, fmt='s', vmin=-1, vmax=1, cmap=cmap,
                    annot_kws={"fontsize": 6})
    plt.savefig(FIG_PATH + f"{name}_{modality}_cor_mat.pdf", transparent=True)
    clear()


def disp_mi_mat(name, modality, order):
    mip_mat = io.loadmat(MI_MAT)
    mis, pvals = mip_mat['mis'], mip_mat['pvals']
    inds = [feature_mapping[fet_name] for fet_name in order]
    mis = mis[inds][:, inds]
    pvals = pvals[inds][:, inds]

    annotations = np.where(pvals < 0.05, '*', '')
    annotations = np.where(pvals < 0.01, '**', annotations)
    annotations = np.where(pvals < 0.001, '***', annotations)

    mi_mat = pd.DataFrame(data=mis, index=order, columns=order)
    mask = np.triu(np.ones_like(mi_mat, dtype=bool))
    vmax = mi_mat.to_numpy()[~mask].max()

    mi_mat = mi_mat.loc[order, order]

    mask = np.triu(np.ones_like(mi_mat, dtype=bool))
    plt.yticks(rotation=30)
    light_red = [0.66080672, 0.21526712, 0.23069468]  # This is the shade of red used in the vlag palette
    cmap = sns.light_palette(light_red, as_cmap=True)
    _ = sns.heatmap(mi_mat, annot=annotations, mask=mask, fmt='s', vmin=0, vmax=vmax, cmap=cmap,
                    annot_kws={"fontsize": 6})
    plt.savefig(FIG_PATH + f"{name}_{modality}_mi_mat.pdf", transparent=True)
    clear()


def cor_x_mi(df, name, mods=None):
    mi_mat = io.loadmat(MI_MAT)['mis']
    corr_mat = np.absolute(df.corr(method="spearman").to_numpy())
    mat = {'mi': [], 'corr': []}

    if mods is None:
        mods = [('spatial', SPATIAL[:-1]), ('waveform', WAVEFORM[:-1]), ('spike-timing', SPIKE_TIMING[:-1])]

    for modality, inds in mods:
        mi_m = mi_mat[inds][:, inds]
        corr_m = corr_mat[inds][:, inds]

        mask = np.triu(np.ones_like(mi_m, dtype=bool))
        mi_m = mi_m[~mask].flatten()
        corr_m = corr_m[~mask].flatten()

        mat['mi'].append(mi_m)
        mat['corr'].append(corr_m)

        r, _ = stats.spearmanr(mi_m, corr_m)
        print(f'spearman r for {modality} is {r: .3f}')

        ax = sns.regplot(x=corr_m, y=mi_m, ci=False)
        ax.set_xlabel('Correlation coefficient (absolute)')
        ax.set_xlim(0, 1.1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_ylabel('Mutual information [bit]')
        ax.set_yticks([0, 0.5, 1, 1.5])

        textstr = f"Spearman's r={r: .3f}\nR^2={r ** 2: .3f}\np="

        # place a text box in upper left in axes coords
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=6)

        plt.savefig(FIG_PATH + f"{name}_{modality}_corr_x_mi.pdf", transparent=True)
        clear()

    io.savemat(STATS_PATH + 'mi_x_corr.mat', mat)


def cdf_plots(df, d_features, name, values=None):
    if values is None:
        values = [None] * len(d_features)
    palette = {"PYR": PYR_COLOR, "PV": PV_COLOR}
    for c in df.columns:
        if c not in d_features:
            continue
        if c in ['TTP_duration']:
            df[c] = 1.6 * df[c] / 256  # transform index to time micro s
        if c in ['FMC_Time-lag_SD', 'SMC_Time-lag_SD']:
            df[c] = 1600 * df[c] / 256
        if c in ['FMC_Time-lag_SS', 'SMC_Time-lag_SS']:
            df[c] = df[c] * ((1.6 / 256) ** 2) * 1000  # transform index to time 1000 * (micros s) ^ 2
        if c in ['FMC_Graph_Average_weight', 'FMC_Graph_Shortest_path']:
            df[c] = 256 * df[c] / 1.6  # transform index to time mm/s

    for c, vals in zip(d_features, values):
        col_pyr = df[c][df.label == 'PYR'].to_numpy()
        col_pv = df[c][df.label == 'PV'].to_numpy()

        pyr_med = np.median(col_pyr)
        pv_med = np.median(col_pv)

        ax_cdf = sns.ecdfplot(data=df, x=c, hue="label", palette=palette)

        if vals is not None:
            ax_cdf.scatter(vals[0], (np.sum(col_pyr < vals[0]) + np.sum(col_pyr <= vals[0])) / (2 * len(col_pyr)),
                           color=PYR_COLOR, s=20)
            ax_cdf.scatter(vals[1], (np.sum(col_pv < vals[1]) + np.sum(col_pv <= vals[1])) / (2 * len(col_pv)),
                           color=PV_COLOR, s=20)

        ax_cdf.axhline(0.5, c='k', alpha=0.3, lw=0.5)
        ax_cdf.axvline(pyr_med, c=PYR_COLOR, lw=2, ls='--')
        ax_cdf.axvline(pv_med, c=PV_COLOR, lw=2, ls='--')

        ax_cdf.set_ylim(ymin=0, ymax=1.1)
        ax_cdf.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax_cdf.set_yticklabels(['0', '', '0.5', '', '1'])
        ax_cdf.spines['top'].set_visible(False)
        ax_cdf.spines['right'].set_visible(False)
        plt.savefig(FIG_PATH + f"{name}_cdf_{c}.pdf", transparent=True)
        clear()


def thr_confusion_mats(preds_path=MAIN_PREDS):
    preds = np.load(preds_path)
    num_chunks = len(CHUNK_SIZES)
    num_modalities = len(MODALITIES)
    mod_ind = [elem[0] for elem in MODALITIES].index('waveform')
    chunk_inds = list(CHUNK_SIZES).index(0)
    preds_0 = preds[chunk_inds::num_chunks]
    preds_wf = preds_0[mod_ind::num_modalities]

    # Numbers were calculated seperately for our data, also possible to get from dataset.
    num_pyr = 83
    num_pv = 21
    labels = np.concatenate((np.ones(num_pyr), np.zeros(num_pv)))

    fig, c_ax = plt.subplots()
    thr = 0.1
    plot_cf_thr(preds_wf, thr, labels, c_ax)
    plt.savefig(FIG_PATH + f"Fig2_B_inset_thr{thr}_WF_conf_mat.pdf", transparent=True)
    clear()

    fig, c_ax = plt.subplots()
    thr = 0.9
    plot_cf_thr(preds_wf, thr, labels, c_ax)
    plt.savefig(FIG_PATH + f"Fig2_B_inset_thr{thr}_WF_conf_mat.pdf", transparent=True)
    clear()


def get_results(modality, name, chunk_size, res_name=MAIN_RES, base_name=BASE_RES):
    results = pd.read_csv(res_name, index_col=0)
    complete = results[results.modality == modality]

    complete = complete[complete.chunk_size.isin([chunk_size])]
    complete = complete.dropna(how='all', axis=1)

    grouped_complete = complete.groupby(by=['modality', 'chunk_size'])

    d = {'spatial': SPATIAL, 'waveform': WAVEFORM, 'spike-timing': SPIKE_TIMING}
    modalities = [(modality, d[modality])]

    if modality == 'spatial':
        base = pd.read_csv(base_name, index_col=0)
        base = base[base.modality == modality]

        base = base[base.chunk_size.isin([chunk_size])]
        base = base.dropna(how='all', axis=1)

        grouped_base = base.groupby(by=['modality', 'chunk_size'])
        plot_fet_imp(grouped_complete.median(), grouped_complete.quantile(0.75), grouped_base.median(),
                     chunk_size=chunk_size, name=f'Fig5_C_{modality}', modalities=modalities,
                     semsn=grouped_complete.quantile(0.25))
        clear()

    chunk_size_list = [chunk_size] if chunk_size == 0 else [0, chunk_size]

    complete_roc = results[results.modality == modality]
    complete_roc = complete_roc[complete_roc.chunk_size.isin(chunk_size_list)]
    complete_roc = complete_roc.dropna(how='all', axis=1)

    plot_roc_curve(complete_roc, name=f'{name}', chunk_size=chunk_size_list, modalities=[modality])
    clear()


def spd(clu_spd, cell_type, name):
    color_main = PYR_COLOR if cell_type == 'pyr' else PV_COLOR
    color_second = LIGHT_PYR if cell_type == 'pyr' else LIGHT_PV
    chunks = clu_spd.calc_mean_waveform()
    if UPSAMPLE != 1:
        chunks = Spike(upsample_spike(chunks.data, UPSAMPLE, 20_000))

    chunks = chunks.get_data()
    chunks = chunks / abs(chunks.min())

    main_chn = get_main(chunks)
    dep = chunks[main_chn].min()
    fig, ax = plt.subplots()
    for i in range(NUM_CHANNELS):
        mean_channel = chunks[i]
        if i == main_chn:
            ax.plot(np.arange(TIMESTEPS * UPSAMPLE), mean_channel, c=color_main)
        else:
            c = color_second if mean_channel.min() <= 0.5 * dep else 'k'
            ax.plot(np.arange(TIMESTEPS * UPSAMPLE), mean_channel, c=c)
    ax.hlines(dep * 0.5, 0, TIMESTEPS * UPSAMPLE - 1, colors='k', linestyles='dashed')

    spd_val = np.sum(chunks.min(axis=1) <= chunks.min() * 0.5)
    print(f"{cell_type} spatial dispersion count is {spd_val}")
    chunk_norm = chunks.min(axis=1) / chunks.min()
    chunk_norm.sort()
    print(f"{cell_type} spatial dispersion sd is {np.std(chunk_norm): .3g}, vector {np.around(chunk_norm, decimals=3)}")

    plt.savefig(FIG_PATH + f"{name}_{cell_type}_spatial_dispersion.pdf", transparent=True)
    clear()

    return spd_val


def fmc_time_lag(clu_fmc_tl, name_fmc_tl, name):
    color_main = PYR_COLOR if name_fmc_tl == 'pyr' else PV_COLOR
    color_second = LIGHT_PYR if name_fmc_tl == 'pyr' else LIGHT_PV

    chunks = clu_fmc_tl.calc_mean_waveform()
    if UPSAMPLE != 1:
        chunks = Spike(upsample_spike(chunks.data, UPSAMPLE, 20_000))

    chunks = chunks.get_data()

    main_c = get_main(chunks)
    amps = chunks.max(axis=1) - chunks.min(axis=1)

    chunks = chunks / (-chunks.min())
    delta = np.zeros(chunks.shape)
    inds = []
    med = np.median(chunks)
    for i in range(len(chunks)):
        sig_m = np.convolve(np.where(chunks[i] <= med, -1, 1), [-1, 1], 'same')
        sig_m[0] = sig_m[-1] = 1
        ind = calc_pos_imp(sig_m, chunks[i].argmin(), DELTA_MODE.F_MCROSS)
        inds.append(ind)
    delta[np.arange(NUM_CHANNELS), inds] = chunks.min(axis=1)
    shift = chunks.shape[-1] // 2 - delta[main_c].argmin()
    cent_delta = np.roll(delta, shift, axis=1)

    dep_inds = delta.argmin(axis=1)
    cent_dep_inds = cent_delta.argmin(axis=1)
    fzc_sd = np.std(((cent_dep_inds - 128) * (1600 / 256))[amps >= 0.25 * amps.max()])
    print(f"{name_fmc_tl} FMC time lag SD value is {fzc_sd: .3g}")

    micro = '\u03BC'

    fig, ax = plt.subplots(NUM_CHANNELS, 1, sharex=True, sharey=True, figsize=(6, 9))
    valid = amps >= 0.25 * amps[main_c]
    window_l = abs((-dep_inds[main_c] + dep_inds[valid]).min()) + 1
    window_r = abs((-dep_inds[main_c] + dep_inds[valid]).max()) + 1
    window = window_l + window_r
    for i, c in enumerate(chunks):
        color = 'k' if not valid[i] else (color_main if i == main_c else color_second)
        i_ax = NUM_CHANNELS - i - 1
        ax[i_ax].plot(c[dep_inds[main_c] - window_l: dep_inds[main_c] + window_r], color=color, alpha=0.5)
        ax[i_ax].axhline(y=med, xmin=0, xmax=2 * window, c='k', alpha=0.3, lw=0.5)
        ax[i_ax].axvline(x=window_l, ymin=-1, ymax=1, color='k', linestyle='--')
        ax[i_ax].axvline(x=window_l - dep_inds[main_c] + dep_inds[i], ymin=-1, ymax=1, color=color)
        ax[i_ax].axis('off')
        ax[i_ax].set_xlim(0, window)
        if valid[i]:
            ax[i_ax].annotate(f"d={'X' if not valid[i] else 1.6e3 * (-dep_inds[main_c] + dep_inds[i]) / 256} {micro}s",
                              xy=(window * 0.75, 0))

    plt.savefig(FIG_PATH + f"{name}_{name_fmc_tl}_fmc_time.pdf", transparent=True)
    clear()

    return fzc_sd


def event_type_comparison_graph(df, d_fets, y_axis, name, conv_func):
    conversion = conv_func(1.6 / 256)
    labels = df.label.to_numpy()
    fmc = df[d_fets['FMC']].to_numpy() * conversion
    smc = df[d_fets['SMC']].to_numpy() * conversion
    neg = df[d_fets['NEG']].to_numpy() * conversion
    fig, ax = plt.subplots()
    xs = np.asarray([1, 2, 3])
    ax.set_ylabel(y_axis)
    ax.set_xlabel('Event')
    counter = 0
    x_shift = 0.1
    ax.set_xticks(xs)
    ax.set_xticklabels(['FMC', 'NEG', 'SMC'])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for label, marker, c in [('PV', '--o', PV_COLOR), ('PYR', '--v', PYR_COLOR)]:
        inds = labels == label
        fmc_temp, smc_temp, neg_temp = fmc[inds], smc[inds], neg[inds]
        stacked = np.vstack((fmc_temp, neg_temp, smc_temp))
        meds = np.median(stacked, axis=1)
        q25 = np.quantile(stacked, 0.25, axis=1)
        q75 = np.quantile(stacked, 0.75, axis=1)
        ax.errorbar(xs + x_shift * counter - 0.5 * x_shift, meds, yerr=[meds - q25, q75 - meds], fmt=marker, c=c)
        counter += 1

    plt.savefig(FIG_PATH + f"{name}_event-type_comparison.pdf", transparent=True)
    clear()


def graph_vals(clu, clu_delta, cell_type):
    chunks = clu.calc_mean_waveform()
    if UPSAMPLE != 1:
        chunks = Spike(upsample_spike(chunks.data, UPSAMPLE, 20_000))

    chunks = chunks.get_data()

    coordinates = COORDINATES
    dists = calculate_distances_matrix(coordinates)

    amp = chunks.max(axis=1) - chunks.min(axis=1)

    arr = clu_delta
    threshold = 0.25 * amp.max()  # Setting the threshold to be self.thr the size of max depolarization

    g_temp = []
    for i in range(NUM_CHANNELS):
        max_dep_index = arr[i].argmin()
        if amp[i] >= threshold:
            g_temp.append((i, max_dep_index))
    g_temp.sort(key=lambda x: x[1])
    assert len(g_temp) > 0

    velocities = []
    for i, (channel1, timestep1) in enumerate(g_temp):
        for channel2, timestep2 in g_temp[i + 1:]:
            if timestep2 != timestep1:
                velocity = dists[channel1, channel2] / (1.6 * (timestep2 - timestep1) / 256)
                print(f"{cell_type} edge weight between channel {channel1} ({timestep1}) and channel {channel2} "
                      f"({timestep2}) is {velocity: .3g}")
                velocities.append(velocity)

    avg_vel = np.mean(velocities)
    print(f"{cell_type} average edge weight is {avg_vel: .3g}")
    return avg_vel


def ach(bin_range, name, clu, bins_in_ms=1, save_fig=True):
    N = 2 * bin_range * bins_in_ms + 1
    offset = 1 / (2 * bins_in_ms)
    bins = np.linspace(-bin_range - offset, bin_range + offset, N + 1)
    c = PV_COLOR if 'pv' in name else PYR_COLOR

    try:
        hist = np.load(FIG_PATH + f'./ach_{name}_{bin_range}_{bins_in_ms}.npy')
    except FileNotFoundError:
        chunks = np.array([np.arange(len(clu.timings))])
        hist = calc_temporal_histogram(clu.timings, bins, chunks)[0]

    np.save(FIG_PATH + f'ach_{name}_{bin_range}_{bins_in_ms}.npy', hist)

    bin_inds = np.convolve(bins, [0.5, 0.5], mode='valid')

    if save_fig:
        fig, ax = plt.subplots()
        ax.bar(bin_inds, hist, color=c, width=bins[1] - bins[0])
        ax.set_ylim(0, 60)
        plt.savefig(FIG_PATH + f"{name}_ACH_{bin_range}_{bins_in_ms}.pdf", transparent=True)
        clear()

    return hist


def unif_dist(hist, name):
    c = PV_COLOR if 'pv' in name else PYR_COLOR
    zero_bin_ind = len(hist) // 2
    hist = (hist[:zero_bin_ind + 1:][::-1] + hist[zero_bin_ind:]) / 2
    hist_up = signal.resample_poly(hist, UPSAMPLE ** 2, UPSAMPLE, padtype='line')
    hist_up = np.where(hist_up >= 0, hist_up, 0)
    cdf = np.cumsum(hist_up) / np.sum(hist_up)

    fig, ax = plt.subplots()
    ax.bar(np.arange(len(cdf)) - (UPSAMPLE // 2), cdf, color=c, width=1)
    lin = np.linspace(cdf[0], cdf[-1], len(cdf))
    dists = abs(lin - cdf)
    ax.plot(np.arange(len(cdf)) - (UPSAMPLE // 2), lin, c='k')
    ax.vlines((np.arange(len(cdf)) - (UPSAMPLE // 2))[::UPSAMPLE], np.minimum(lin, cdf)[::UPSAMPLE],
              np.maximum(lin, cdf)[::UPSAMPLE],
              colors='k', linestyles='dashed')
    ax.set_ylim(ymin=0, ymax=1.1)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticks(np.asarray([0, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400]) * 2)
    ret = np.sum(dists) / len(cdf)
    print(f"{name} Uniform-distance is {ret: .3g}")
    plt.savefig(FIG_PATH + f"{name}_unif_dist.pdf", transparent=True)
    clear()

    return ret


def plot_delta(clu, name):
    color_main = PYR_COLOR if name == 'pyr' else PV_COLOR
    color_second = LIGHT_PYR if name == 'pyr' else LIGHT_PV
    chunks = clu.calc_mean_waveform()
    main_c = get_main(chunks.data)
    if UPSAMPLE != 1:
        chunks = Spike(upsample_spike(chunks.data, UPSAMPLE, SAMPLE_RATE))
    chunks = chunks.get_data()
    chunk_amp = chunks.max(axis=1) - chunks.min(axis=1)
    amp_thr = (chunk_amp >= 0.25 * chunk_amp.max())
    chunks = chunks / (-chunks.min())
    delta = np.zeros(chunks.shape)
    inds = []
    med = np.median(chunks)
    for i in range(len(chunks)):
        sig_m = np.convolve(np.where(chunks[i] <= med, -1, 1), [-1, 1], 'same')
        sig_m[0] = sig_m[-1] = 1
        ind = calc_pos_imp(sig_m, chunks[i].argmin(), DELTA_MODE.F_MCROSS)
        inds.append(ind)
    delta[np.arange(NUM_CHANNELS), inds] = chunks.min(axis=1)

    fig, ax = plt.subplots(NUM_CHANNELS, 1, sharex=True, sharey=True, figsize=(3, 20))
    for i, c_ax in enumerate(ax[::-1]):
        c_ax.plot(chunks[i], c=color_second if amp_thr[i] else '#505050')
        c_ax.plot(delta[i], c=color_main if amp_thr[i] else '#505050')
        c_ax.axis('off')

    plt.savefig(FIG_PATH + f"Fig3_Aa_{name}_delta.pdf", transparent=True)
    clear()

    shift = chunks.shape[-1] // 2 - delta[main_c].argmin()
    ret = np.roll(delta, shift, axis=1)

    fig, ax = plt.subplots(NUM_CHANNELS, 1, sharex=True, sharey=True, figsize=(6, 20))
    for i, c_ax in enumerate(ax[::-1]):
        c_ax.plot(delta[i], c=color_second if amp_thr[i] else '#505050')
        c_ax.plot(ret[i], c=color_main if amp_thr[i] else '#505050')
        c_ax.annotate(text='', xy=(delta[i].argmin(), delta[i].min() / 2), xytext=(ret[i].argmin(), ret[i].min() / 2),
                      arrowprops=dict(arrowstyle='<-'))

        c_ax.axis('off')

    plt.savefig(FIG_PATH + f"Fig3_Ab_{name}_delta_cent.pdf", transparent=True)
    clear()

    return ret


def get_delta_results(modality, chunk_size=(0, ), preds_path=TRANS_WF_PREDS, res_path=TRANS_WF_RES):
    results = pd.read_csv(res_path, index_col=0)
    complete = results[results.modality == modality]
    complete = complete.dropna(how='all', axis=1)

    preds = np.load(preds_path)

    # Numbers were calculated seperately for our data, also possible to get from dataset.
    num_pyr = 83
    num_pv = 21
    labels = np.concatenate((np.ones(num_pyr), np.zeros(num_pv)))

    fig, c_ax = plt.subplots()
    plot_cf_thr(preds, 0.5, labels, c_ax)
    plt.savefig(FIG_PATH + f"Fig3_C_inset_transformed_wf_confusion_matrix.pdf", transparent=True)
    clear()

    plot_roc_curve(complete, name='Fig3_C_transformed_wf', chunk_size=chunk_size, modalities=[modality])
    clear()


def calc_pos(arr, start_pos):
    pos = start_pos
    while pos >= 0:
        if arr[pos] == 0:
            pos -= 1
        else:
            break
    return pos


def match_spike(channel, med):
    pos = channel.argmin()
    spike = np.zeros(channel.shape)
    sig_m = np.convolve(np.where(channel <= med, -1, 1), [-1, 1], 'same')
    sig_m[0] = sig_m[-1] = 1
    pos = calc_pos(sig_m, pos)
    spike[pos] = channel.min()

    return spike, pos


def match_chunk(wf, amp):
    ret = np.zeros(wf.shape)
    main_c = amp.argmax()
    roll_val = 0
    med = np.median(wf)
    for i, channel in enumerate(wf):
        ret[i], pos = match_spike(channel, med)
        if i == main_c:
            roll_val = wf.shape[-1] // 2 - pos

    ret = np.roll(ret, roll_val, axis=1)
    chunk = ret / abs(ret.min())

    return chunk


def wf_heatmap(labels, path=SAVE_PATH_WF_TRANS + 'mean_spikes/'):
    waveforms = []
    files = [path + f for f in listdir(path) if isfile(join(path, f))]
    for f in files:
        waveforms.append(np.load(f))

    pv_inds = np.argwhere(labels == 'PV').flatten()
    pyr_inds = np.argwhere(labels == 'PYR').flatten()

    if UPSAMPLE != 1:
        waveforms_up = np.asarray([upsample_spike(wf, UPSAMPLE, SAMPLE_RATE) for wf in waveforms])
    else:
        waveforms_up = waveforms

    main_channels = np.asarray([wf[get_main(wf)] for wf in waveforms_up])
    main_channels = main_channels / np.expand_dims(abs(main_channels.min(axis=1)), axis=1)

    reg_pv_order = pv_inds[np.argsort(main_channels[pv_inds].argmin(axis=1))]
    reg_pyr_order = pyr_inds[np.argsort(main_channels[pyr_inds].argmin(axis=1))]
    reg_inds = np.concatenate((reg_pv_order, reg_pyr_order))
    reg_sorted_wf = main_channels[reg_inds]

    fig, ax = plt.subplots()
    shw = ax.imshow(reg_sorted_wf, vmin=-1, vmax=reg_sorted_wf.max(), cmap='gray', interpolation='none')
    ax.hlines(y=len(pv_inds), xmin=0, xmax=256, color='k')
    plt.colorbar(shw)
    plt.savefig(FIG_PATH + f"Fig3_B_left_heatmap_org.pdf", transparent=True)

    trans_wfs = []
    for wf in waveforms_up:
        amp = wf.max(axis=1) - wf.min(axis=1)
        wavelets_fzc = match_chunk(wf, amp)
        trans_wfs.append(wavelets_fzc)
    trans_main_channels = np.asarray([wf[get_main(wf)] for wf in trans_wfs])

    trans_pv_order = pv_inds[np.argsort(trans_main_channels[pv_inds].argmin(axis=1))]
    trans_pyr_order = pyr_inds[np.argsort(trans_main_channels[pyr_inds].argmin(axis=1))]
    trans_inds = np.concatenate((trans_pv_order, trans_pyr_order))
    trans_sorted_wf = trans_main_channels[trans_inds]

    fig, ax = plt.subplots()
    shw = ax.imshow(trans_sorted_wf, vmin=-1, vmax=reg_sorted_wf.max(), cmap='gray', interpolation='none')
    ax.hlines(y=len(pv_inds), xmin=0, xmax=256, color='k')
    plt.colorbar(shw)

    plt.savefig(FIG_PATH + f"Fig3_B_right_heatmap_trans.pdf", transparent=True)
    clear()


def chunk_results(name, res_path=MAIN_RES):
    results = pd.read_csv(res_path, index_col=0)

    drop = [col for col in results.columns.values if col not in ['modality', 'chunk_size', 'seed', 'auc']]

    results = results.drop(columns=drop)

    plot_auc_chunks_bp(results, name=f'{name}_chunks_rf', mods=['spatial'])
    clear()


def get_comp_results(chunk_sizes=(BEST_WF_CHUNK, BEST_ST_CHUNK, BEST_SPATIAL_CHUNK),
                     res_path=REGION_CA1_RES, train_name='CA1', res_path2=REGION_NCX_RES, train_name2='NCX'):
    results = pd.read_csv(res_path, index_col=0)

    complete = results[results.chunk_size.isin(chunk_sizes)]

    results2 = pd.read_csv(res_path2, index_col=0)

    complete2 = results2[results2.chunk_size.isin(chunk_sizes)]

    for mod_name, cs in [('spatial', BEST_SPATIAL_CHUNK), ('spike-timing', BEST_ST_CHUNK), ('waveform', BEST_WF_CHUNK)]:
        # ROC curves
        plot_roc_curve(complete, name=f'Fig6-1_{train_name}_NCX', chunk_size=[cs], use_alt=False, modalities=[mod_name])
        clear()
        plot_roc_curve(complete, name=f'Fig6-1_{train_name}_CA1', chunk_size=[cs], use_alt=True, modalities=[mod_name])
        clear()
        plot_roc_curve(complete2, name=f'Fig6-1_{train_name2}_CA1', chunk_size=[cs], use_alt=False,
                       modalities=[mod_name])
        clear()
        plot_roc_curve(complete2, name=f'Fig6-1_{train_name2}_NCX', chunk_size=[cs], use_alt=True,
                       modalities=[mod_name])
        clear()

    #  AUC
    # chunk sizes correspond to waveform, spike-timing spatial
    plot_test_vs_test2_bp(complete, chunk_sizes=(BEST_WF_CHUNK, BEST_ST_CHUNK, BEST_SPATIAL_CHUNK),
                          name=f'Fig6_A_ncx_vs_ca1_auc', df2=complete2)
    clear()
    plot_test_vs_test2_bp(complete, chunk_sizes=(BEST_WF_CHUNK, BEST_ST_CHUNK, BEST_SPATIAL_CHUNK),
                          name=f'Fig6_B_ncx_vs_ca1_auc', diff=True, df2=complete2)
    clear()

    grouped_complete = complete.groupby(by=['modality', 'chunk_size'])
    grouped_complete2 = complete2.groupby(by=['modality', 'chunk_size'])

    modalities = [('spatial', SPATIAL)]
    plot_fet_imp(grouped_complete.median(), grouped_complete.quantile(0.75), None,
                 chunk_size=BEST_SPATIAL_CHUNK, name=f'Fig6-2A_{train_name}_spatial', modalities=modalities,
                 semsn=grouped_complete.quantile(0.25))
    clear()
    plot_fet_imp(grouped_complete2.median(), grouped_complete2.quantile(0.75), None,
                 chunk_size=BEST_SPATIAL_CHUNK, name=f'Fig6-2B_{train_name2}_spatial', modalities=modalities,
                 semsn=grouped_complete2.quantile(0.25))
    clear()


def spike_trains(pyr_cluster, pv_cluster):
    pyr_mask = (pyr_cluster.timings > 3509050) * (pyr_cluster.timings < 3510050)
    pv_mask = (pv_cluster.timings > 3509050) * (pv_cluster.timings < 3510050)
    pyr_train = pyr_cluster.timings[pyr_mask]
    pv_train = pv_cluster.timings[pv_mask]

    fig, ax = plt.subplots(figsize=(10, 1))
    ax.axis('off')
    ax.vlines(pyr_train, 0.05, 0.45, colors=PYR_COLOR)
    ax.vlines(pv_train, 0.55, 0.95, colors=PV_COLOR)
    scalebar = AnchoredSizeBar(ax.transData, 200, '200 ms', 'lower right', pad=0.1, color='k', frameon=False,
                               size_vertical=0.01)

    ax.add_artist(scalebar)
    plt.savefig(FIG_PATH + "Fig1_Ab_bottom_spike_train.pdf", transparent=True)
    clear()


def costume_imp(modality, chunk_size, res_path, base_path, imp_inds, name_mapping, name):
    results = pd.read_csv(res_path, index_col=0)
    complete = results[results.modality == modality]

    complete = complete[complete.chunk_size.isin([chunk_size])]
    complete = complete.dropna(how='all', axis=1)

    grouped_complete = complete.groupby(by=['modality', 'chunk_size'])

    base = pd.read_csv(base_path, index_col=0)
    base = base[base.modality == modality]

    base = base[base.chunk_size.isin([chunk_size])]
    base = base.dropna(how='all', axis=1)

    grouped_base = base.groupby(by=['modality', 'chunk_size'])

    modalities = [(modality, imp_inds)]

    plot_fet_imp(grouped_complete.median(), grouped_complete.quantile(0.75), grouped_base.median(),
                 chunk_size=chunk_size, name=f'{name}_{modality}', modalities=modalities,
                 semsn=grouped_complete.quantile(0.25), fet_names_map=name_mapping)
    clear()


def calc_aw(a, b):
    a, b = a.copy(), b.copy()
    a.sort()
    b.sort()

    na = len(a)
    nb = len(b)

    smaller = np.median(a) < np.median(b)
    aw = 0
    for v in a:
        if smaller:
            aw_v = (v < b).sum() + 0.5 * (b == v).sum()
        else:
            aw_v = (v > b).sum() + 0.5 * (b == v).sum()
        aw += aw_v / nb
        if (aw_v / nb) > 1:
            raise AssertionError
    return aw / na


def spatial_var(path):
    arr = []
    labels = []
    files = listdir(path)
    df = None

    for file in sorted(files):
        if df is None:
            df = pd.read_csv(path + '/' + file)
        else:
            temp = pd.read_csv(path + '/' + file)
            df = df.append(temp)
    df = np.array(df.to_numpy()[:, SPATIAL[:-1]], dtype=np.float64)
    scaler = StandardScaler()
    scaler.fit(df)

    df_temp = None
    for file in sorted(files):
        df_temp = pd.read_csv(path + '/' + file)
        df_np = np.array(df_temp.to_numpy()[:, SPATIAL[:-1]], dtype=np.float64)
        df_np = scaler.transform(df_np)

        std = np.std(df_np, axis=0)
        arr.append(std)
        label = df_temp.label.to_numpy()[0]
        labels.append(int(label))

    labels = np.asarray(labels)
    columns = df_temp.columns.values[SPATIAL[:-1]]

    arr_m = np.asarray(arr)[:, SPATIAL[:-1]]
    arr_pyr = arr_m[labels == 1]
    arr_pv = arr_m[labels == 0]

    arr_pyr_med = np.median(arr_pyr, axis=0)
    arr_pv_med = np.median(arr_pv, axis=0)
    p = stats.wilcoxon(arr_pyr_med, arr_pv_med).pvalue
    print('PYR within unit SD', np.around(np.quantile(arr_pyr_med, q=[0.25, 0.5, 0.75]), decimals=3))
    print('PV within unit SD', np.around(np.quantile(arr_pv_med, q=[0.25, 0.5, 0.75]), decimals=3))
    print(f'Wilcoxon for PYR compared to PV (coupled features): p-val={p}')

    fig, ax = plt.subplots(figsize=(6, 10))
    for pyr, pv, pyr_all, pv_all, col in zip(arr_pyr_med, arr_pv_med, arr_pyr.T, arr_pv.T, columns):
        p = stats.mannwhitneyu(pyr_all, pv_all).pvalue
        print(f"{col}: p-val={p: .3g}; effect size={calc_aw(pyr_all, pv_all): .3g}")
        print(f"    PYR: {np.around(np.quantile(pyr_all, q=[0.25, 0.5, 0.75]), decimals=3)}")
        print(f"    PV: {np.around(np.quantile(pv_all, q=[0.25, 0.5, 0.75]), decimals=3)}")
        ax.plot([pyr, pv], 'k' if p < 0.05 else '0.7', marker='o')

    yerr = [[np.median(arr_pyr_med) - np.quantile(arr_pyr_med, 0.25),
             np.median(arr_pv_med) - np.quantile(arr_pv_med, 0.25)],
            [np.quantile(arr_pyr_med, 0.75) - np.median(arr_pyr_med),
             np.quantile(arr_pv_med, 0.75) - np.median(arr_pv_med)]]
    ax.bar([0, 1], [np.median(arr_pyr_med), np.median(arr_pv_med)], color='#A0A0A0', tick_label=['PYR', 'PV'],
           yerr=yerr)

    plt.savefig(FIG_PATH + "Fig4_B_spatial_var.pdf", transparent=True)
    clear()


def create_figs():
    if not isdir(FIG_PATH):
        mkdir(FIG_PATH)

    # Load wanted units - no light
    pyr_cluster = load_cluster(DATA_TEMP_PATH, PYR_NAME)
    pv_cluster = load_cluster(DATA_TEMP_PATH, PV_NAME)
    trans_units(pyr_cluster)
    trans_units(pv_cluster)

    # Figure 1
    print('Creating Figure 1')
    # Fig. 1Aa - PSTH:
    pv_cluster_full = load_cluster(TEMP_PATH_FULL, PV_NAME)
    disp_psth(pv_cluster_full, DATA_PATH)
    plt.savefig(FIG_PATH + "Fig1_Aa_bottom_PSTH.pdf", transparent=True)
    clear()
    # Fig. 1Ab (top) - LFP was created manually using neuroscope from dat files
    # Fig. 1Ab (bottom) - Spike trains:
    spike_trains(pyr_cluster, pv_cluster)
    # Fig. 1B (top) - Waveforms:
    pyr_cluster.plot_cluster(save=True, path=FIG_PATH, name='Fig1_B_top_waveform')
    clear()
    pv_cluster.plot_cluster(save=True, path=FIG_PATH, name='Fig1_B_top_waveform')
    clear()
    # Fig. 1B (middle) - ACH:
    _ = ach(30, 'Fig1_B_middle_pv', pv_cluster)
    _ = ach(30, 'Fig1_B_middle_pyr', pyr_cluster)
    # Fig. 1B (bottom) - CCH:
    pyr_cluster_full = load_cluster(TEMP_PATH_FULL, PYR_NAME)
    disp_cch(pyr_cluster_full, pv_cluster_full)
    plt.savefig(FIG_PATH + "Fig1_B_bottom_CCH.pdf", transparent=True)
    clear()
    # Fig. 1C - Pie chart:
    df = load_df(['region', 'label'], trans_labels=False)
    pie_chart(df)

    # Figure 2
    print('\nCreating Figure 2')
    # Fig. 2Aa - Waveform and TTP-duration:
    pyr_ttp = TTP_dur(pyr_cluster, PYR_COLOR, 'pyr')
    pv_ttp = TTP_dur(pv_cluster, PV_COLOR, 'pv')
    # Fig. 2Ab - TTP-duration CDF plot:
    df = load_df()
    d_features = ['TTP_duration']
    cdf_plots(df, d_features, 'Fig2_Ab', [[pyr_ttp, pv_ttp]])
    # Fig. 2B - waveform ROC curves:
    get_results('waveform', 'Fig2_B', BEST_WF_CHUNK)
    # Fig. 2B (inset) - confusion matrices:
    thr_confusion_mats()
    # Fig. 2Ca - Uniform-distance:
    hist_pv = ach(50, 'pv_no_light', pv_cluster, bins_in_ms=2, save_fig=False)
    hist_pyr = ach(50, 'pyr_no_light', pyr_cluster, bins_in_ms=2, save_fig=False)
    pv_unif_dist = unif_dist(hist_pv, 'Fig2_Ca_pv')
    pyr_unif_dist = unif_dist(hist_pyr, 'Fig2_Ca_pyr')
    # Fig. 2Cb - Uniform-distance CDF plot:
    d_features = ['Uniform_distance']
    cdf_plots(df, d_features, 'Fig2_Cb_spike-timing', [[pyr_unif_dist, pv_unif_dist]])
    # Fig. 2D - spike-timing ROC curves:
    get_results('spike-timing', 'Fig2_D', BEST_ST_CHUNK)

    # Figure 2 - Extended Data
    # Fig. 2-1A - waveform correlation matrix:
    order_wf = ['TTP_duration', 'TTP_magnitude', 'FWHM', 'Rise_coefficient', 'Max_speed', 'Break_measure', 'Smile_cry',
                'Acceleration']
    disp_corr_mat(df, 'Fig2-1_A', 'wf', order_wf)
    # Fig. 2-1B - waveform MI matrix:
    disp_mi_mat('Fig2-1_B', 'wf', order_wf)
    # Fig. 2-1B (inset) - waveform correlation between CC and MI:
    cor_x_mi(df, 'Fig2-1_B_inset', [('wf', WAVEFORM[:-1])])
    # Fig. 2-2B - spike-timing correlation matrix:
    order_st = ['Uniform_distance', 'D_KL_short', 'Rise_time', 'Jump_index', 'D_KL_long', 'PSD_center', "PSD'_center",
                'Firing_rate']
    disp_corr_mat(df, 'Fig2-2_A', 'st', order_st)
    # Fig. 2-2B - spike-timing MI matrix:
    disp_mi_mat('Fig2-2_B', 'st', order_st)
    # Fig. 2-2B (inset) - spike-timing correlation between CC and MI:
    cor_x_mi(df, 'Fig2-2_B_inset', [('st', SPIKE_TIMING[:-1])])

    # Figure 3
    print('\nCreating Figure 3')
    # Fig. 3Aa and 3Ab - Event-based delta-transformation:
    pyr_fmc_delta = plot_delta(pyr_cluster, 'pyr')
    pv_fmc_delta = plot_delta(pv_cluster, 'pv')
    # Fig. 3B - pre- post-rtansformation comparison
    wf_heatmap(df.label.to_numpy())
    # Fig. 3C - transformed spikes classification results
    get_delta_results('trans_wf')
    # Fig. 3Da - FMC-Time-lag-SD illustrations:
    pyr_fmc_sd = fmc_time_lag(pyr_cluster, 'pyr', 'Fig3_Da')
    pv_fmc_sd = fmc_time_lag(pv_cluster, 'pv', 'Fig3_Da')
    # Fig. 3Da - FMC-Time-lag-SD CDF:
    cdf_plots(df, ['FMC_Time-lag_SD'], 'Fig3_Db', [[pyr_fmc_sd, pv_fmc_sd]])
    # Fig. 3Ea - FMC_Graph_Average_weight edge weigths:
    pyr_graph_avg = graph_vals(pyr_cluster, pyr_fmc_delta, 'pyr')
    pv_graph_avg = graph_vals(pv_cluster, pv_fmc_delta, 'pv')
    # Fig. 3Ea - FMC_Graph_Average_weight CDF:
    cdf_plots(df, ['FMC_Graph_Average_weight'], 'Fig3_Eb', [[pyr_graph_avg, pv_graph_avg]])
    # Fig. 3Fa - SPD-Count illustrations:
    pyr_spd = spd(pyr_cluster, 'pyr', 'Fig3_Fa')
    pv_spd = spd(pv_cluster, 'pv', 'Fig3_Fa')
    # Fig. 3Fa - SPD-Count CDF:
    cdf_plots(df, ['SPD_Count'], 'Fig3_Fb', [[pyr_spd, pv_spd]])

    # Figure 3 - Extended Data
    # Fig. 3-1A - spatial correlation matrix:
    order_spatial = ["NEG_Time-lag_SS", "NEG_Time-lag_SD", "FMC_Time-lag_SS", "FMC_Time-lag_SD", "SMC_Time-lag_SS",
                     "SMC_Time-lag_SD", "NEG_Graph_Average_weight", "NEG_Graph_Shortest_path", 'NEG_Graph_Longest_path',
                     "FMC_Graph_Average_weight", "FMC_Graph_Shortest_path", 'FMC_Graph_Longest_path',
                     "SMC_Graph_Average_weight", "SMC_Graph_Shortest_path", 'SMC_Graph_Longest_path', 'SPD_Count',
                     'SPD_SD', 'SPD_Area']
    disp_corr_mat(df, 'Fig3-1_A', 'spatial', order_spatial)
    # Fig. 3-1B - spatial MI matrix:
    disp_mi_mat('Fig3-1_B', 'spatial', order_spatial)
    # Fig. 3-1B (inset) - spatial correlation between CC and MI:
    cor_x_mi(df, 'Fig3-1_B_inset', [('spatial', SPATIAL[:-1])])

    # Figure 4
    print('\nCreating Figure 4')
    # Fig. 4A - Event-based and cell type differences in Time-lag-SD:
    event_type_comparison_graph(df, {'FMC': 'FMC_Time-lag_SD', 'NEG': 'NEG_Time-lag_SD', 'SMC': 'SMC_Time-lag_SD'},
                                'SD [micro-s]', 'Fig4_A_time_lag_SD', lambda x: 1000 * x)
    # Fig. 4B - within-chunk SD comparison:
    spatial_var(f'{SAVE_PATH}{BEST_SPATIAL_CHUNK}')

    # Figure 4 - Extended Data
    # Fig 4-1A - Event-based and cell type differences in Time-lag-SS:
    event_type_comparison_graph(df, {'FMC': 'FMC_Time-lag_SS', 'NEG': 'NEG_Time-lag_SS', 'SMC': 'SMC_Time-lag_SS'},
                                'Time-lag-SS [ms^2]', 'Fig4-1_A_time_lag_SS', lambda x: x ** 2)
    event_type_comparison_graph(df, {'FMC': 'FMC_Graph_Shortest_path', 'NEG': 'NEG_Graph_Shortest_path',
                                     'SMC': 'SMC_Graph_Shortest_path'}, 'Shortest-[ath [mm/s]',
                                'Fig4-1_B_Shortest_path', lambda x: 1 / x)

    # Figure 5
    print('\nCreating Figure 5')
    # Fig 5A - Chunk comparison:
    chunk_results('Fig5_A')
    # Fig 5B and 5C - ROC curve and feature importance:
    get_results('spatial', 'Fig5_B', BEST_SPATIAL_CHUNK)
    # Fig 5C (inset) - event importance:
    imp_inds = np.arange(NUM_EVENTS + 1)  # extra +1 as a workaround for usual case where last value is -1
    name = 'Fig5_c_inset_events'
    costume_imp('spatial', BEST_SPATIAL_CHUNK, EVENTS_RES, BASE_EVENTS_RES, imp_inds, ['FMC', 'NEG', 'SMC'], name)

    # Figure 5 - Extended Data:
    # Fig 5-1A-F - CDFs for most important features
    important_features = ['FMC_Time-lag_SS', 'FMC_Time-lag_SD', 'SMC_Time-lag_SS', 'SMC_Time-lag_SD',
                          'FMC_Graph_Average_weight', 'FMC_Graph_Shortest_path']
    cdf_plots(df, important_features, 'Fig5-1_spatial')
    # Figure 6
    print('\nCreating Figure 6')
    get_comp_results()


if __name__ == '__main__':
    create_figs()
