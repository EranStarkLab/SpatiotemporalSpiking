import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import interpolate
from sklearn.metrics import confusion_matrix

from constants import SPATIAL, SPIKE_TIMING, WAVEFORM
from constants import feature_names_org as FET_NAMES
from paths import FIG_PATH

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

NUM_FETS = 34

def change_length(x, y, length):
    """
    Update length of two input arrays so that x has no repeating values (exception for the last value)
    """
    valid = []
    for i in range(len(x)):
        if i == len(x) - 1:
            valid.append(i)
        elif x[i] != x[i + 1]:
            valid.append(i)
        else:
            continue

    x = x[valid]
    y = y[valid]

    xnew = np.linspace(0, 1, length)
    f = interpolate.interp1d(x, y, kind='linear')
    ynew = [0] + list(f(xnew))
    return np.array(ynew)


def str2lst(s):
    """
    Converts a string representing a list to a list
    """
    ret = []
    for val in list(s[1:-1].split(" ")):  # remove [] and split
        if len(val) == 0:
            continue
        if val[-1] == '.':  # for 0. case
            val = val[:-1]
        ret.append(float(val))
    return np.array(ret)


def plot_roc_curve(df, name=None, chunk_size=(200, ), modalities=None, use_alt=False):
    """
    Creates ROC curve plot based on the modalities and chunk size given as input. Multiple chunk sizes can be viewed
    together
    """
    if modalities is None:
        modalities = ['spatial', 'spike-timing', 'waveform']
    mean_fprs = None

    for m_name in modalities:
        df_m = df[df.modality == m_name]
        fig, ax = plt.subplots()
        for cz in chunk_size:
            df_cz = df_m[df_m.chunk_size == cz]

            fpr_col = 'fpr' if not use_alt else 'fpr2'
            tpr_col = 'tpr' if not use_alt else 'tpr2'

            fprs = [str2lst(lst) for lst in df_cz[fpr_col]]
            tprs = [str2lst(lst) for lst in df_cz[tpr_col]]

            x_length = 100

            tprs = np.array([change_length(fpr, tpr, x_length) for tpr, fpr in zip(tprs, fprs)])

            mean_fprs = [0] + list(np.linspace(0, 1, x_length))
            med_tprs = np.median(tprs, axis=0)
            iqr25 = np.quantile(tprs, 0.25, axis=0)
            iqr75 = np.quantile(tprs, 0.75, axis=0)

            auc_col = 'auc' if not use_alt else 'auc2'

            auc = np.median(df_cz[auc_col])

            label_add = f", chunk size={cz}" if len(chunk_size) > 1 else ""

            ax.plot(mean_fprs, med_tprs, label=f"AUC={auc:.3f}" + label_add)
            ax.fill_between(mean_fprs, iqr25, iqr75, alpha=0.2)

        ax.plot(mean_fprs, mean_fprs, color='k', linestyle='--', label='chance level')
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.legend()
        ax.set_xlabel("False PYR rate")
        ax.set_ylabel("True PYR rate")

        if name is None:
            plt.title(f"{m_name} chunk sizes: {chunk_size}")
            plt.show()
        else:
            plt.savefig(FIG_PATH + f"{name}_{m_name}_roc_curve.pdf", transparent=True)


def plot_cf_thr(preds: np.ndarray, thr, labels, c_ax):
    """
    Plots a confusion matrix based on threshold thr
    """
    preds_thr = preds >= thr
    confs_thr = [confusion_matrix(labels, pred) for pred in preds_thr]
    conf_thr = np.roll(np.roll(np.median(confs_thr, axis=0), 1, axis=0), 1, axis=1)

    data = conf_thr / np.expand_dims(conf_thr.sum(axis=1), axis=1)
    labels = [[f"{conf_thr[0, 0]:2g}", f"{conf_thr[0, 1]:2g}"],
              [f"{conf_thr[1, 0]:2g}", f"{conf_thr[1, 1]:2g}"]]
    cmap = sns.light_palette("seagreen", as_cmap=True)
    ticklabels = ['PYR', 'PV']
    _ = sns.heatmap(data, annot=labels, fmt='', vmin=0, vmax=1, cmap=cmap, ax=c_ax, xticklabels=ticklabels,
                    yticklabels=ticklabels)
    cbar = c_ax.collections[0].colorbar
    cbar.set_ticks([0, .5, 1])
    cbar.set_ticklabels(['0%', '50%', '100%'])
    c_ax.set_ylabel("True label")
    c_ax.set_xlabel('Predicted label')

    return c_ax


def plot_fet_imp(df, sems, base, name=None, chunk_size=0, modalities=None, semsn=None, fet_names_map=None):
    """
    Creates a plot of feature importance
    Parameters
    ----------
    df : Medians to plot
    sems : information for (positive if semsn is not None) error bars
    base : Baseline information
    name : Name used for saving the plot
    chunk_size : Chunk size to plot
    modalities : Modalities to plot, both name and feature indices are required
    semsn : information for negative error bars
    fet_names_map : Feature names to add to plot

    Returns
    -------
    None. Plot is saved
    """
    if modalities is None:
        modalities = [('spatial', SPATIAL), ('spike-timing', SPIKE_TIMING), ('waveform', WAVEFORM)]

    if fet_names_map is None:
        fet_names_map = FET_NAMES

    fets_org = [f"test feature {f + 1}" for f in range(NUM_FETS)]
    rem = [f"test2 feature {f + 1}" for f in range(NUM_FETS) if f"test2 feature {f + 1}" in df.columns]
    df = df.drop(columns=rem)
    sems = sems.drop(columns=rem)
    base = None if base is None else base.drop(columns=rem)
    if semsn is not None:
        semsn = semsn.drop(columns=rem)

    m = {f: name for (f, name) in zip(fets_org, fet_names_map)}
    df = df.rename(m, axis='columns')
    df = df.drop(columns=['seed', 'auc', 'auc2'], errors='ignore')
    sems = sems.rename(m, axis='columns')
    sems = sems.drop(columns=['seed', 'auc', 'auc2'], errors='ignore')
    if base is not None:
        base = base.rename(m, axis='columns')
        base = base.drop(columns=['seed', 'auc', 'auc2'], errors='ignore')

    if semsn is not None:
        semsn = semsn.rename(m, axis='columns')
        semsn = semsn.drop(columns=['seed', 'auc', 'auc2'], errors='ignore')

    for m_name, m_places in modalities:
        df_m = df.xs((m_name, chunk_size), level=["modality", "chunk_size"]).dropna(axis=1).to_numpy()
        sems_m = sems.xs((m_name, chunk_size), level=["modality", "chunk_size"]).dropna(axis=1).to_numpy()
        base_m = None if base is None else base.xs((m_name, chunk_size), level=["modality", "chunk_size"]).dropna(axis=1).to_numpy()
        semsn_m = None
        if semsn is not None:
            semsn_m = semsn.xs((m_name, chunk_size), level=["modality", "chunk_size"]).dropna(axis=1).to_numpy()

        names_m = np.asarray(fet_names_map)[m_places[:-1]]

        df_order = np.asarray(df_m)
        order = np.argsort((-1 * df_order).mean(axis=0))

        names_m = names_m[order]

        x = np.arange(len(names_m))  # the label locations

        width = 0.75  # the width of the bars

        fig, ax = plt.subplots(figsize=(12, 6))

        color = '#A0A0A0'

        df_cm = df_m.flatten()[order]
        sems_cm = sems_m.flatten()[order]
        base_cm = None if base is None else base_m.flatten()[order]
        if semsn_m is not None:
            semsn_cm = 2 * df_cm - semsn_m.flatten()[order]
            sems_cm = np.concatenate((np.expand_dims(semsn_cm, axis=0), np.expand_dims(sems_cm, axis=0))) - df_cm
        else:
            sems_cz = np.concatenate((np.expand_dims(sems_cz, axis=0), np.expand_dims(sems_cz, axis=0)))

        ax.bar(x, df_cm, width, yerr=sems_cm, color=color)
        if base is not None:
            ax.hlines(base_cm, x - (width / 2), x + (width / 2), linestyles='dashed', color='k')
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Importance (SHAP value)')
        ax.set_xticks(x)
        ax.set_xticklabels(names_m, rotation=-90)

        fig.tight_layout()

        if name is None:
            name = ''
        else:
            plt.savefig(FIG_PATH + f"{name}_fet_imp.pdf", transparent=True)


def plot_auc_chunks_bp(df, name=None, plot=True, ax_inp=None, edge_color='k', shift=0, mods=None):
    """
    Plots box plot for the AUC for each chunk size for a single modality
    """
    chunk_sizes = np.roll(df.chunk_size.unique(), -1)[::-1]
    if mods is None:
        mods = df.modality.unique()

    for m_name in mods:
        df_m = df[df.modality == m_name]
        chunk_sizes_m = df_m.chunk_size.to_numpy()
        chunk_aucs = [df_m.auc[chunk_sizes_m == cs].to_numpy() for cs in chunk_sizes]

        if ax_inp is None:
            fig, ax = plt.subplots(figsize=(9, 6))
        else:
            ax = ax_inp

        ax.axhline(y=np.median(df_m.auc[chunk_sizes_m == 0].to_numpy()), color='k', linestyle='--')
        bp = ax.boxplot(chunk_aucs, labels=chunk_sizes.astype(np.int32),
                        positions=2.5 * np.arange(len(chunk_sizes)) + shift,
                        flierprops=dict(markeredgecolor=edge_color, marker='+'), notch=True, bootstrap=1_000)

        for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], color=edge_color)

        ax.set_xticks(2.5 * np.arange(len(chunk_sizes)))

        ax.set_ylabel('AUC')
        ax.set_xlabel('Chunk Size')
        ax.set_ylim(top=1.01)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if not plot:
            return ax

        if name is None:
            plt.show()
        else:
            plt.savefig(FIG_PATH + f"{name}_{m_name}_auc.pdf", transparent=True)
            plt.clf()
            plt.cla()
            plt.close('all')


def plot_test_vs_test2_bp(df, chunk_sizes=(0, 0, 0), name=None, diff=False, df2=None):
    """
    Plots a comparative box plot for the CA1-trained and neocortical-trained models for both test sets
    """
    labels = ['waveform', 'spike-timing', 'spatial']
    figsize = (13, 7.5) if not diff else (6, 7.5)
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(labels)) + 1  # the label locations
    width = 0.4

    for i, (mod, cz) in enumerate(zip(labels, chunk_sizes)):
        filter1 = df.chunk_size == cz
        filter2 = df.modality == mod

        val = df[filter1 & filter2].auc.to_numpy()
        test2_val = df[filter1 & filter2].auc2.to_numpy()

        val2, test2_val2 = None, None
        if df2 is not None:
            filter1 = df2.chunk_size == cz
            filter2 = df2.modality == mod

            val2 = df2[filter1 & filter2].auc.to_numpy()
            test2_val2 = df2[filter1 & filter2].auc2.to_numpy()

        if diff:
            positions = [x[i]] if df2 is None else [x[i] - width * 0.63]
            diff_vals = 100 * (test2_val - val) / test2_val
            ax.boxplot(diff_vals, positions=positions, boxprops={"facecolor": "k"},
                       flierprops=dict(markerfacecolor='#808080', marker='+'), medianprops={"color": "k"},
                       patch_artist=True, widths=width, notch=True, bootstrap=1_000)
            if df2 is not None:
                positions = [x[i] + width * 0.63]
                diff_vals = 100 * (test2_val2 - val2) / test2_val2
                ax.boxplot(diff_vals, positions=positions, boxprops={"facecolor": "r"},
                           flierprops=dict(markerfacecolor='#808080', marker='+'), medianprops={"color": "k"},
                           patch_artist=True, widths=width, notch=True, bootstrap=1_000)
        else:
            position = [x[i] - width * 0.63]
            ax.boxplot(val, positions=position, boxprops={"facecolor": "k"},
                       flierprops=dict(markerfacecolor='#808080', marker='+'), medianprops={"color": "k"},
                       patch_artist=True, widths=width, notch=True, bootstrap=1_000)
            position = [x[i] + width * 0.63]
            ax.boxplot(test2_val, positions=position, boxprops={"facecolor": "k"},
                       flierprops=dict(markerfacecolor='#808080', marker='+'), medianprops={"color": "k"},
                       patch_artist=True, widths=width, notch=True, bootstrap=1_000)
            if df2 is not None:
                ax.boxplot(test2_val2, positions=[x[i] + len(x) + 0.3 - width * 0.63], boxprops={"facecolor": "r"},
                           flierprops=dict(markerfacecolor='#808080', marker='+'), medianprops={"color": "k"},
                           patch_artist=True, widths=width, notch=True, bootstrap=1_000)
                ax.boxplot(val2, positions=[x[i] + len(x) + 0.3 + width * 0.63], boxprops={"facecolor": "r"},
                           flierprops=dict(markerfacecolor='#808080', marker='+'), medianprops={"color": "k"},
                           patch_artist=True, widths=width, notch=True, bootstrap=1_000)

    if diff:
        ticks = x
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
    else:
        ticks = []
        tick_labels = []
        for t in x:
            ticks += [t - width * 0.63, t + width * 0.63]
            tick_labels += ['nCX', 'CA1']
        for t in x:
            ticks += [t + len(x) + 0.3 - width * 0.63, t + len(x) + 0.3 + width * 0.63]
            tick_labels += ['nCX', 'CA1']
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    if name is None:
        plt.show()
    else:
        diff_str = "diff" if diff else ""
        plt.savefig(FIG_PATH + f"{name}{diff_str}.pdf", transparent=True)
