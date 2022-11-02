import ml.ML_util as ML_util

import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import auc as auc_func
from sklearn.preprocessing import StandardScaler
import shap

from ml.gs_rf import grid_search as grid_search_rf
from ml.RF import run as run_model

from constants import CHUNK_SIZES, CHUNKS_MAP, MODALITIES, TRAIN_PER, TEST_PER
from constants import NUM_ITER, SKIP_0_IMP, SKIP_TEST_IMP, SKIP_TEST2_IMP
from constants import RUN_NAME, LOAD_ITER, REGION_BASED, TRAIN_CA1, SHUFFLE, GS_ZERO, SHAP_SAMPLE
from constants import n_estimators_min, n_estimators_max, n_estimators_num, max_depth_min, max_depth_max, max_depth_num
from constants import min_samples_splits_min, min_samples_splits_max, min_samples_splits_num
from constants import min_samples_leafs_min, min_samples_leafs_max, min_samples_leafs_num
from constants import INF, N
from paths import ML_INPUT_PATH, DATASET_PATH, RESULTS_PATH

from ml.hideen_prints import HiddenPrints

NUM_FETS = sum([len(modal[1]) - 1 for modal in MODALITIES])  # Total number of features in data (all modalities)
dataset_identifier = f'{TRAIN_PER}0{TEST_PER}'


def get_test_set(data_path, get_alt=False):
    """

    Parameters
    ----------
    data_path : String; path to dataset
    get_alt : Bool; return the second tests set

    Returns
    -------
    Scaled tests set features and test set labels
    """
    train, test2, test1, _, _, _ = ML_util.get_dataset(data_path)

    train_squeezed = ML_util.squeeze_clusters(train)
    train_data = train_squeezed
    features, labels = ML_util.split_features(train_data)
    features = np.nan_to_num(features)
    features = np.clip(features, -INF, INF)

    scaler = StandardScaler()
    scaler.fit(features)

    test_set = test1 if not get_alt else test2
    if len(test_set) == 0:
        return [], []

    test_squeezed = ML_util.squeeze_clusters(test_set)
    x, y = ML_util.split_features(test_squeezed)
    x = np.nan_to_num(x)
    x = np.clip(x, -INF, INF)
    x = scaler.transform(x)

    return x, y


def get_shap_imp(clf, test, seed, skip=False):
    """

    Parameters
    ----------
    clf : Classifier to use for SHAP calculation
    test : Test set
    seed : Seed used for sampling of the test set
    skip : Bool; do not perform analysis and return zeros with a similar shape to the expected output

    Returns
    -------
    Matrix of SHAP values for each sample and feature
    """
    if skip:
        a, b = np.zeros((test.shape[-1])), np.zeros((min(SHAP_SAMPLE, len(test)), test.shape[-1]))
        return a, b
    df = pd.DataFrame(test)
    df_shap = df.sample(min(SHAP_SAMPLE, len(test)), random_state=int(seed) + 1)
    rf_explainer = shap.TreeExplainer(clf)  # define explainer
    # calculate shap values; check_aditivity set to overcome a possible bug in the shap library
    shap_values = rf_explainer(df_shap, check_additivity=False)
    pyr_shap_values = shap_values[..., 1]
    return np.mean(np.abs(pyr_shap_values.values), axis=0), pyr_shap_values.values


def get_predictions(clf, train, test2, test1, use_alt):
    """

    Parameters
    ----------
    clf : Classifier to use prediction
    train : Training set, used for scaling of test data
    test2 : Test set of same region data if REGION_BASED=True
    test1 : Test set (if REGION_BASED=True contains data from the non-trained-pon region)
    use_alt : Bool; which test set to use. False->test1

    Returns
    -------
    Prediction of the classifier for every sample in the test set
    """
    train_squeezed = ML_util.squeeze_clusters(train)
    train_features, train_labels = ML_util.split_features(train_squeezed)
    train_features = np.nan_to_num(train_features)
    train_features = np.clip(train_features, -INF, INF)

    scaler = StandardScaler()
    scaler.fit(train_features)

    preds = []

    test_set = test1 if not use_alt else test2
    if len(test_set) == 0:
        return None

    for cluster in test_set:
        features, labels = ML_util.split_features(cluster)
        features = np.nan_to_num(features)
        features = np.clip(features, -INF, INF)
        features = scaler.transform(features)

        prob = clf.predict_proba(features).mean(axis=0)
        pred = prob[1]

        preds.append(pred)

    return np.asarray(preds)


def get_preds(clf, data_path):
    """

    Parameters
    ----------
    clf : Classifier to use prediction
    data_path : String; path to data

    Returns
    -------
    Prediction of the classifier for both test sets
    """
    train, test2, test1, _, _, _ = ML_util.get_dataset(data_path)

    test1_preds = get_predictions(clf, train, test2, test1, False)
    assert test1_preds is not None
    test2_preds = get_predictions(clf, train, test2, test1, True)
    if test2_preds is None:
        test2_preds = []

    return test1_preds, test2_preds


def calc_auc(clf, data_path, use_alt=False):
    """

    Parameters
    ----------
    clf : Classifier used for prediction
    data_path : String; path to dataset
    use_alt : Bool; which test set to use. False->test1

    Returns
    -------
    AUC for the model on the test set, lists of false positive and true positive rate later used for plotting
    """
    train, test2, test1, _, _, _ = ML_util.get_dataset(data_path)

    test_set = test1 if not use_alt else test2
    if len(test_set) == 0:
        return 0, [0], [0]

    targets = [row[0][-1] for row in test_set]

    preds = get_predictions(clf, train, test2, test1, use_alt)
    assert preds is not None

    # calculate fpr and tpr values for different thresholds
    fpr, tpr, thresholds = roc_curve(targets, preds, drop_intermediate=False)
    auc_val = auc_func(fpr, tpr)
    return auc_val, fpr, tpr


def get_modality_results(data_path, seed, fet_inds, region_based=False, shuffle_labels=False, gs_zero=True):
    """
    Get results for a single seed X modality combination
    Parameters
    ----------
    data_path : String; path to dataset
    seed : Seed used for model training and partitioning into folds in grid search
    fet_inds : indices of the features of the modality
    region_based : Bool; True iff training on a single region
    shuffle_labels : Bool; whether to shuffle labels
    gs_zero : Bool; whether to choose hyperparameters based on grid search of the no-chunking model

    Returns
    -------
    dataftrame of the performance and importance and matrices of the raw predictions and feature importance values
    """
    aucs, fprs, tprs, importances = [], [], [], []
    aucs2, fprs2, tprs2, importances2 = [], [], [], []
    preds, preds2 = [], []
    raw_imps, raw_imps2 = [], []

    n_estimators, max_depth, min_samples_split, min_samples_leaf = None, None, None, None

    if gs_zero:
        print(f"            Starting chunk size = 0")

        clf, n_estimators, max_depth, min_samples_split, min_samples_leaf = grid_search_rf(
            data_path + f"/0_{dataset_identifier}/", n_estimators_min,
            n_estimators_max, n_estimators_num, max_depth_min, max_depth_max, max_depth_num,
            min_samples_splits_min, min_samples_splits_max, min_samples_splits_num,
            min_samples_leafs_min, min_samples_leafs_max, min_samples_leafs_num, N, seed=seed,
            region_based=region_based, shuffle_labels=shuffle_labels)

        pred, pred2 = get_preds(clf, data_path + f"/0_{dataset_identifier}/")

        auc, fpr, tpr = calc_auc(clf, data_path + f"/0_{dataset_identifier}/")
        auc2, fpr2, tpr2 = calc_auc(clf, data_path + f"/0_{dataset_identifier}/", use_alt=True)

        print(f"\nchunk size: 0 - AUC: {auc}, test2 AUC: {auc2}\n")

        x, _ = get_test_set(data_path + f"/0_{dataset_identifier}/")
        raw_imp = np.ones((SHAP_SAMPLE, NUM_FETS)) * np.nan
        importance, raw_imp_temp = get_shap_imp(clf, x, seed, skip=SKIP_0_IMP or SKIP_TEST_IMP)
        raw_imp[:min(SHAP_SAMPLE, len(x)), fet_inds[:-1]] = raw_imp_temp

        x, _ = get_test_set(data_path + f"/0_{dataset_identifier}/", get_alt=True)
        raw_imp2 = np.ones((SHAP_SAMPLE, NUM_FETS)) * np.nan
        if len(x) > 0:
            importance2, raw_imp_temp2 = get_shap_imp(clf, x, seed, skip=SKIP_0_IMP or SKIP_TEST2_IMP)
            raw_imp2[:min(SHAP_SAMPLE, len(x)), fet_inds[:-1]] = raw_imp_temp2
        else:
            importance2 = np.ones(importance.shape) * np.nan

        aucs.append(auc)
        fprs.append(fpr)
        tprs.append(tpr)
        importances.append(importance)
        aucs2.append(auc2)
        fprs2.append(fpr2)
        tprs2.append(tpr2)
        importances2.append(importance2)
        preds.append(pred)
        preds2.append(pred2)
        raw_imps.append(raw_imp)
        raw_imps2.append(raw_imp2)

    modality = data_path.split('/')[-1]

    chunks_iter = list(CHUNK_SIZES) if CHUNKS_MAP is None else CHUNKS_MAP[modality]
    for chunk_size in chunks_iter:
        if chunk_size == 0 and gs_zero:
            continue
        print(f"            Starting chunk size = {chunk_size}")
        if gs_zero:
            clf = run_model(n_estimators, max_depth, min_samples_split, min_samples_leaf,
                            data_path + f"/{chunk_size}_{dataset_identifier}/", seed, shuffle_labels)
        else:
            clf, _, _, _, _ = grid_search_rf(data_path + f"/{chunk_size}_{dataset_identifier}/", n_estimators_min,
                                             n_estimators_max, n_estimators_num, max_depth_min, max_depth_max,
                                             max_depth_num, min_samples_splits_min, min_samples_splits_max,
                                             min_samples_splits_num, min_samples_leafs_min, min_samples_leafs_max,
                                             min_samples_leafs_num, N, seed=seed, region_based=region_based,
                                             shuffle_labels=shuffle_labels)

        auc, fpr, tpr = calc_auc(clf, data_path + f"/{chunk_size}_{dataset_identifier}/")
        auc2, fpr2, tpr2 = calc_auc(clf, data_path + f"/{chunk_size}_{dataset_identifier}/", use_alt=True)

        print(f"\nchunk size: {chunk_size} - AUC: {auc}, test2 AUC: {auc2}\n")

        pred, pred2 = get_preds(clf, data_path + f"/{chunk_size}_{dataset_identifier}/")

        x, _ = get_test_set(data_path + f"/{chunk_size}_{dataset_identifier}/")
        raw_imp = np.ones((SHAP_SAMPLE, NUM_FETS)) * np.nan
        importance, raw_imp_temp = get_shap_imp(clf, x, seed, skip=SKIP_TEST_IMP)
        raw_imp[:min(SHAP_SAMPLE, len(x)), fet_inds[:-1]] = raw_imp_temp

        x, _ = get_test_set(data_path + f"/{chunk_size}_{dataset_identifier}/", get_alt=True)
        raw_imp2 = np.ones((SHAP_SAMPLE, NUM_FETS)) * np.nan
        if len(x) > 0:
            importance2, raw_imp_temp2 = get_shap_imp(clf, x, seed, skip=SKIP_TEST2_IMP)
            raw_imp2[:min(SHAP_SAMPLE, len(x)), fet_inds[:-1]] = raw_imp_temp2
        else:
            importance2 = np.ones(importance.shape) * np.nan

        aucs.append(auc)
        fprs.append(fpr)
        tprs.append(tpr)
        importances.append(importance)
        aucs2.append(auc2)
        fprs2.append(fpr2)
        tprs2.append(tpr2)
        importances2.append(importance2)
        preds.append(pred)
        preds2.append(pred2)
        raw_imps.append(raw_imp)
        raw_imps2.append(raw_imp2)

    chunk_size_df = chunks_iter
    if gs_zero and 0 not in chunks_iter:
        chunk_size_df = [0] + chunk_size_df

    df = pd.DataFrame(
        {'modality': modality, 'chunk_size': chunk_size_df, 'seed': [str(seed)] * len(aucs),
         'auc': aucs, 'fpr': fprs, 'tpr': tprs, 'auc2': aucs2, 'fpr2': fprs2, 'tpr2': tprs2})

    features = [f"test feature {f + 1}" for f in range(NUM_FETS)]
    importances_row = np.nan * np.ones((len(aucs), NUM_FETS))
    importances_row[:, fet_inds[:-1]] = importances
    df[features] = pd.DataFrame(importances_row, index=df.index)

    features = [f"test2 feature {f + 1}" for f in range(NUM_FETS)]
    importances_row = np.nan * np.ones((len(aucs), NUM_FETS))
    importances_row[:, fet_inds[:-1]] = importances2
    df[features] = pd.DataFrame(importances_row, index=df.index)

    return df, np.stack(tuple(preds)), np.stack(tuple(preds2)), np.stack(tuple(raw_imps)), np.stack(tuple(raw_imps2))


def get_folder_results(data_path, seed, region_based=False, shuffle_labels=0, gs_zero=True):
    """
    get results for a single seed
    Parameters
    ----------
    data_path : String; path to dataset
    seed : Seed used for model training and partitioning into folds in grid search
    region_based : Bool; True iff training on a single region
    shuffle_labels : Bool; whether to shuffle labels
    gs_zero : Bool; whether to choose hyperparameters based on grid search of the no-chunking model

    Returns
    -------

    """
    df_cols = ['modality', 'chunk_size', 'seed', 'auc', 'fpr', 'tpr', 'auc2', 'fpr2', 'tpr2'] + \
              [f"test feature {f + 1}" for f in range(NUM_FETS)] + [f"test2 feature {f + 1}" for f in range(NUM_FETS)]
    preds, preds2 = None, None
    raw_imps, raw_imps2 = None, None
    df = pd.DataFrame({col: [] for col in df_cols})
    for modality in MODALITIES:
        print(f"        Starting modality {modality[0]}")
        if shuffle_labels > 0:
            for i in np.arange(shuffle_labels):
                print(f"        Starting shuffle iteration {i}")
                new_seed = seed * shuffle_labels + i
                modality_df, pred, pred2, raw_imp, raw_imp2 = get_modality_results(
                    data_path + '/' + modality[0], new_seed, modality[1], region_based=region_based,
                    shuffle_labels=True, gs_zero=gs_zero)
                df = df.append(modality_df, ignore_index=True)
                preds = pred if preds is None else np.vstack((preds, pred))
                preds2 = pred2 if preds2 is None else np.vstack((preds2, pred2))
                raw_imps = raw_imp if raw_imps is None else np.vstack((raw_imps, raw_imp))
                raw_imps2 = raw_imp2 if raw_imps2 is None else np.vstack((raw_imps2, raw_imp2))
        else:
            modality_df, pred, pred2, raw_imp, raw_imp2 = get_modality_results(
                data_path + '/' + modality[0], seed, modality[1], region_based=region_based,
                shuffle_labels=False, gs_zero=gs_zero)
            df = df.append(modality_df, ignore_index=True)
            preds = pred if preds is None else np.vstack((preds, pred))
            preds2 = pred2 if preds2 is None else np.vstack((preds2, pred2))
            raw_imps = raw_imp if raw_imps is None else np.vstack((raw_imps, raw_imp))
            raw_imps2 = raw_imp2 if raw_imps2 is None else np.vstack((raw_imps2, raw_imp2))

    return df, preds, preds2, raw_imps, raw_imps2


def main():
    """
    Create datasets and tests model performance
    Returns
    -------
    None. All outputs are saved as csv and npy files.
    """
    results_path = RESULTS_PATH + f'data_sets_{RUN_NAME}/'
    if not os.path.isdir(results_path):
        os.mkdir(results_path)

    results = None if not LOAD_ITER else pd.read_csv(results_path + f'results_rf_{RUN_NAME}_temp.csv', index_col=0)
    preds = None if not LOAD_ITER else np.load(results_path + f'preds_rf_{RUN_NAME}_temp.npy')
    raw_imps = None if not LOAD_ITER else np.load(results_path + f'raw_imps_rf_{RUN_NAME}_temp.npy')
    preds2 = None
    raw_imps2 = None
    if REGION_BASED and LOAD_ITER:
        preds2 = np.load(results_path + f'preds2_rf_{RUN_NAME}_temp.npy')
        raw_imps2 = np.load(results_path + f'raw_imps2_rf_{RUN_NAME}_temp.npy')

    iter_num = 0 if not LOAD_ITER else len(results) / (len(CHUNK_SIZES) * len(MODALITIES))

    dataset_path = DATASET_PATH + f'data_sets_{RUN_NAME}'
    if not os.path.isdir(dataset_path):
        os.mkdir(dataset_path)

    for i in range(iter_num, NUM_ITER):
        print(f"Starting iteration {i}")
        new_path = dataset_path + f"/seed_{i}"
        if not os.path.isdir(new_path):
            os.mkdir(new_path)
        for name, places in MODALITIES:
            new_new_path = new_path + f"/{name}/"
            if not os.path.isdir(new_new_path):
                os.mkdir(new_new_path)
            keep = places
            with HiddenPrints():
                postprocessed_data_path = ML_INPUT_PATH
                ML_util.create_datasets(TRAIN_PER, 0, TEST_PER, postprocessed_data_path, CHUNK_SIZES, new_new_path,
                                        keep, verbos=False, seed=i, region_based=REGION_BASED, train_ca1=TRAIN_CA1)

        result, pred, pred2, raw_imp, raw_imp2 = get_folder_results(new_path, i, REGION_BASED, shuffle_labels=SHUFFLE,
                                                                    gs_zero=GS_ZERO)
        if results is None:
            results = result
        else:
            results = results.append(result)

        preds = pred if preds is None else np.vstack((preds, pred))
        raw_imps = raw_imp if raw_imps is None else np.vstack((raw_imps, raw_imp))

        np.save(results_path + f'preds_rf_{RUN_NAME}_temp', preds)
        np.save(results_path + f'raw_imps_rf_{RUN_NAME}_temp', raw_imps)

        if REGION_BASED:
            preds2 = pred2 if preds2 is None else np.vstack((preds2, pred2))
            raw_imps = raw_imp2 if raw_imps2 is None else np.vstack((raw_imps2, raw_imp2))
            np.save(results_path + f'preds2_rf_{RUN_NAME}_temp', preds2)
            np.save(results_path + f'raw_imps2_rf_{RUN_NAME}_temp', raw_imps2)
        results.to_csv(results_path + f'results_rf_{RUN_NAME}_temp.csv')

    results.to_csv(results_path + f'results_rf_{RUN_NAME}.csv')
    np.save(results_path + f'preds_rf_{RUN_NAME}', preds)
    np.save(results_path + f'raw_imps_rf_{RUN_NAME}', raw_imps)
    if REGION_BASED:
        np.save(results_path + f'preds2_rf_{RUN_NAME}', preds2)
        np.save(results_path + f'raw_imps2_rf_{RUN_NAME}', raw_imps2)


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

    if not os.path.isdir(DATASET_PATH):
        os.mkdir(DATASET_PATH)

    if not os.path.isdir(RESULTS_PATH):
        os.mkdir(RESULTS_PATH)

    main()
