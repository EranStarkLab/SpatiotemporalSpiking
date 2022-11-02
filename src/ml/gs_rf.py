from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import time
from constants import INF, SAMPLE_N

import ml.ML_util as ML_util

def get_inds_unsq(cum_c, inds):
    """

    ----------
    cum_c : cumulative number of samples per unit
    inds : set indices

    Returns
    -------
    Indices in the flat version of the data
    """
    pairs = []
    for i in inds:
        if i == 0:
            pairs.append((0, cum_c[i]))
        else:
            pairs.append((cum_c[i - 1], cum_c[i]))

    new_inds_raw = [np.arange(pair[0], pair[1]) for pair in pairs]
    new_inds = np.concatenate(new_inds_raw)
    return new_inds

def cv_gen(unsqueezed, n, seed):
    """

    Parameters
    ----------
    unsqueezed : np.ndarray; unsqueezed full data
    n : number of folds
    seed : seed for randomization

    Returns
    -------
    List of Indices for random stratified partition
    """
    ret = []
    sham_data = np.asarray([elem[0] for elem in unsqueezed])
    sham_features, labels = ML_util.split_features(sham_data)
    cum_c = np.cumsum([len(elem) for elem in unsqueezed])
    basic_split = StratifiedKFold(n_splits=n, shuffle=True, random_state=seed)
    i = 1
    for inds_train, inds_test in basic_split.split(sham_features, labels):
        new_inds_train_full = get_inds_unsq(cum_c, inds_train)
        new_inds_test = get_inds_unsq(cum_c, inds_test)

        np.random.seed(seed * n + i)
        length = len(new_inds_train_full)
        inds = np.random.choice(length, min(SAMPLE_N, length), replace=False)
        new_inds_train_sample = new_inds_train_full[inds]

        ret.append((new_inds_train_sample, new_inds_test))
        i += 1

    return ret

def grid_search(dataset_path, n_estimators_min, n_estimators_max, n_estimators_num,
                max_depth_min, max_depth_max, max_depth_num, min_samples_splits_min, min_samples_splits_max,
                min_samples_splits_num, min_samples_leafs_min, min_samples_leafs_max, min_samples_leafs_num, n,
                seed=0, region_based=False, shuffle_labels=False):
    """

    Parameters
    ----------
    dataset_path : String; path to dataset
    n : Number of folds
    seed : Seed used for randomization
    region_based : Bool; perform analysis on a single region, used for some assertion (not necessary if not following
    the process described in the paper
    shuffle_labels : Bool; whether to shuffle labels

    Returns
    -------
    A trained classifier and best hyperparameters
    """
    train, _, test1, _, _, _ = ML_util.get_dataset(dataset_path)

    train_squeezed = ML_util.squeeze_clusters(train)
    train_data = train_squeezed
    features, labels = ML_util.split_features(train_data)
    features = np.nan_to_num(features)
    features = np.clip(features, -INF, INF)

    if shuffle_labels:
        np.random.seed(seed)
        np.random.shuffle(labels)

    scaler = StandardScaler()
    scaler.fit(features)
    features = scaler.transform(features)

    cv = StratifiedKFold(n_splits=n, shuffle=True, random_state=seed)
    if len(features) > SAMPLE_N:
        assert region_based
        cv = cv_gen(train, n, seed)

    n_estimatorss = np.logspace(n_estimators_min, n_estimators_max, n_estimators_num).astype('int')
    max_depths = np.logspace(max_depth_min, max_depth_max, max_depth_num).astype('int')
    min_samples_splits = np.logspace(min_samples_splits_min, min_samples_splits_max, min_samples_splits_num,
                                     base=2).astype('int')
    min_samples_leafs = np.logspace(min_samples_leafs_min, min_samples_leafs_max, min_samples_leafs_num, base=2).astype(
        'int')

    print()
    parameters = {'n_estimators': n_estimatorss, 'max_depth': max_depths, 'min_samples_split': min_samples_splits,
                  'min_samples_leaf': min_samples_leafs}
    model = RandomForestClassifier(random_state=seed, class_weight='balanced')
    clf = GridSearchCV(model, parameters, cv=cv, verbose=0,
                       scoring='roc_auc')
    print('Starting grid search...')
    start = time.time()
    clf.fit(features, labels)
    end = time.time()
    print('Grid search completed in %.2f seconds, best parameters are:' % (end - start))
    print(clf.best_params_)

    n_estimators = clf.best_params_['n_estimators']
    max_depth = clf.best_params_['max_depth']
    min_samples_split = clf.best_params_['min_samples_split']
    min_samples_leaf = clf.best_params_['min_samples_leaf']
    # need to create another one as the other trains on both train and dev
    classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                        min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                        random_state=seed, class_weight='balanced')
    classifier.fit(features, labels)

    return classifier, n_estimators, max_depth, min_samples_split, min_samples_leaf
