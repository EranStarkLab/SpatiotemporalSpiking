from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import time

import ml.ML_util as ML_util
from constants import INF

def run(n_estimators, max_depth, min_samples_split, min_samples_leaf, dataset_path, seed, shuffle_labels=False):
    """
    runner function for the RF model. see gs_rf.py for explanations about the inputs.
   """

    train, test2, test1, _, _, _ = ML_util.get_dataset(dataset_path)

    train_squeezed = ML_util.squeeze_clusters(train)
    train_features, train_labels = ML_util.split_features(train_squeezed)
    train_features = np.nan_to_num(train_features)
    train_features = np.clip(train_features, -INF, INF)

    if shuffle_labels:
        np.random.seed(seed)
        np.random.shuffle(train_labels)

    print('Scaling data...')
    scaler = StandardScaler()
    scaler.fit(train_features)
    train_features = scaler.transform(train_features)
    clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                 min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                 random_state=seed, class_weight='balanced')
    print('Fitting Random Forest model...')
    start = time.time()
    clf.fit(train_features, train_labels)
    end = time.time()
    print('Fitting took %.2f seconds' % (end - start))

    return clf
