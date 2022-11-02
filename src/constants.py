import numpy as np

# Module 1 - Raw data parsing

# A dictionary determining which sessions will be read and which shanks will be skipped
SESSION_EMPTY_INDS = {'es25nov11_13': [1, 2, 4]}

# The following was used for the entire dataset (not provided)
"""SESSION_EMPTY_INDS = {
    'es04feb12_1': [],
    'es09feb12_2': [1], 'es09feb12_3': [1],
    'es20may12_1': [2, 3], 'es21may12_1': [2, 3, 4],
    'es25nov11_3': [], 'es25nov11_5': [], 'es25nov11_9': [], 'es25nov11_12': [], 'es25nov11_13': [],
    'es27mar12.012': [4], 'es27mar12.013': [3, 4], 'es27mar12_2': [3, 4], 'es27mar12_3': [3, 4],
    'm258r1_7': [], 'm258r1_42': [], 'm258r1_44': [], 'm258r1_48': [1, 2],
    'm361r2_13': [3, 4], 'm361r2_17': [], 'm361r2_20': [], 'm361r2_34': [1, 3], 'm371r2_3': [1, 2, 3],
    'm531r1_10': [], 'm531r1_11': [], 'm531r1_29': [2], 'm531r1_31': [2], 'm531r1_32': [2], 'm531r1_34': [2],
    'm531r1_35': [2], 'm531r1_36': [2], 'm531r1_38': [2], 'm531r1_40': [2], 'm531r1_41': [2], 'm531r1_42': [2],
    'm531r1_43': [2],
    'm649r1_3': [3], 'm649r1_5': [], 'm649r1_14': [2, 3], 'm649r1_16': [1], 'm649r1_17': [1], 'm649r1_19': [1],
    'm649r1_21': [1], 'm649r1_22': [1]
}"""

# Change of the following parameters may lead to unexpected behavior.
# number of channels per shank
NUM_CHANNELS = 8

# Number of samples per spike
TIMESTEPS = 32

# number of bytes to read each time from the spk files, based on the data format of 16 bit integers
NUM_BYTES = 2

# Sampling rate
SAMPLE_RATE = 20_000


#######################################################################################################################
# Module 2 - Feature extraction

# chunk sizes to extract (0 means no chunking)
# Note that with small chunk sizes, RuntimeWarning may be encountered specfically due to sparse ACHs
CHUNK_SIZES = (0, 25, 50, 100, 200, 400, 800, 1600)

# Binary value indicating whether to perform delta-transformation on the waveforms before extracting waveform-based
# features. If True, only waveform-based features will be extracted.
TRANS_WF = False

# Change to True to add extra prints for execution time information
VERBOS = False

# Seed for randomized chunking partition
SEED = 2

# Value to replace infinity when encountered in feature extraction, also used in Module 3
INF = 9999

# Change of the following parameters may lead to unexpected beahvior.
# Upsampling factor for the waveforms and ACHs
UPSAMPLE = 8

# Total number of features per original feature after chunk statistics extraction
RICH_FACTOR = 6

# Coordinates of the electrodes on the shanks
COORDINATES = np.array([[0, 0], [-9, 20], [8, 40], [-13, 60], [12, 80], [-17, 100], [16, 120], [-21, 140]])

# Time window of ACH in ms
ACH_WINDOW = 1000

# Number of bins per ms of ACH
ACH_RESOLUTION = 2


#######################################################################################################################
# Module 3 - Machine learning

# Feature indices in preprocessed data files of spatial features
SPATIAL = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, -1]
SPATIAL_R = np.arange(18 * RICH_FACTOR + 1)  # 18 features
SPATIAL_R[-1] = -1

# Feature indices in preprocessed data files of waveform-based features
WAVEFORM = [18, 19, 20, 21, 22, 23, 24, 25, -1]
WAVEFORM_R = np.arange(8 * RICH_FACTOR + 1) + 18 * RICH_FACTOR  # 18 previous features, 8 new features
WAVEFORM_R[-1] = -1

# Feature indices in preprocessed data files of spike-timing features
SPIKE_TIMING = [26, 27, 28, 29, 30, 31, 32, 33, -1]
SPIKE_TIMING_R = np.arange(8 * RICH_FACTOR + 1) + (18 + 8) * RICH_FACTOR  # 18 + 8 previous features, 8 new features
SPIKE_TIMING_R[-1] = -1

# Feature indices in preprocessed data files of waveform-based features when TRANS_WF is True (see above)
TRANS_WF_IND = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, -1]

# Name identifier of the run, outputs would be saved under this name
RUN_NAME = 'test_run'

# Whether to load and continue from an execution started before using the same RUN_NAME
LOAD_ITER = False

# whether to train only on a single region. Training region is determined by TRAIN_CA1 (see below). When REGION_BASED is
# True, performance on the non-trained-upon region test set would be saved in the auc field whereas performance on the
# training region test set would be saved in the aucs field (similar for other fields)
REGION_BASED = False

# Whether to train the model on CA1 or NCX data. Relevant only when REGION_BASED is True.
TRAIN_CA1 = True

# Whether to shuffle dataset labels to get baseline performance.If 0 or False no shuffling would occurre, otherwise,
# shuffling would occurre SHUFFLE times per train-test partition.
SHUFFLE = 0

# Whether to choose hyperparameters for all chunks based on the no-chunking grid search or to perform grid search for
# every chunk size (slower). If True, grid search on the no-chunking data would be performed even if 0 is not in
# CHUNK_SIZES
GS_ZERO = True

# Number of iterations for every modality x chunk size
NUM_ITER = 50

# Number of grid search folds
N = 5

# Number of samples from the training set used for training models in grid search procedure when GS_ZERO is False
SAMPLE_N = 5_000

# Number of samples from the test set used to calculate SHAP values
SHAP_SAMPLE = 1_000

# Whether to calculate importance for chunk size 0 (False would speed execution)
SKIP_0_IMP = False

# Whether to calculate importance for the test set (i.e., other region when region based; False would speed execution)
SKIP_TEST_IMP = False

# Whether to calculate importance for the test2 set (i.e., same region when region based, otherwise there is no effect;
# False would speed execution)
SKIP_TEST2_IMP = False

# Fraction of samples in each set, development set is not used. Should be summed to 1
TRAIN_PER, DEV_PER, TEST_PER = 0.8, 0, 0.2
assert TRAIN_PER + DEV_PER + TEST_PER == 1

# If using chunk statistics use the following
MODALITIES = [('spatial', SPATIAL_R), ('spike-timing', SPIKE_TIMING_R), ('waveform', WAVEFORM_R)]
# If not using chunk statistics use the following (you may want to change CHUNK_SIZES to test only no chunking).
# MODALITIES = [('spatial', SPATIAL), ('spike-timing', SPIKE_TIMING), ('waveform', WAVEFORM)]
# If testing delta-transformed waveform-based features use the following
# MODALITIES = [('trans_wf', TRANS_WF_IND)]

# If you want to test only specific chunk sizes per modality use this
CHUNKS_MAP = None
# CHUNKS_MAP = {'spatial': [25], 'spike-timing': [1_600], 'waveform': [50]}  # 0 is included implicitly unless GS_ZERO
# is False

# Hyperparameters tested in the grid search. See sklearn documentation for their effect. Note that the numbers here
# span the range tested on an exponential scale
n_estimators_min = 0  # base 10
n_estimators_max = 2
n_estimators_num = 3
max_depth_min = 1  # base 10
max_depth_max = 2
max_depth_num = 2
min_samples_splits_min = 1  # base 2
min_samples_splits_max = 5
min_samples_splits_num = 5
min_samples_leafs_min = 0  # base 2
min_samples_leafs_max = 5
min_samples_leafs_num = 6

# Feature (no metadata) names according to the order in the output of Module 2
# Feature names are used for feature importance calculation based on differnt groupings
feature_names_org = ["SPD_Count", "SPD_SD", "SPD_Area", "NEG_Time-lag_SS", "NEG_Time-lag_SD", "FMC_Time-lag_SS",
                     "FMC_Time-lag_SD", "SMC_Time-lag_SS", "SMC_Time-lag_SD", "NEG_Graph_Average_weight",
                     "NEG_Graph_Shortest_path", "NEG_Graph_Longest_path", "FMC_Graph_Average_weight",
                     "FMC_Graph_Shortest_path", "FMC_Graph_Longest_path", "SMC_Graph_Average_weight",
                     "SMC_Graph_Shortest_path", "SMC_Graph_Longest_path", "Break_measure", "FWHM", "Acceleration",
                     "Max_speed", "TTP_magnitude", 'TTP_duration', "Rise_coefficient", "Smile_cry", "Firing_rate",
                     "D_KL_short", "D_KL_long", "Jump_index", "PSD_center", "PSD'_center", "Rise_time",
                     "Uniform_distance"]

feature_names_rich = []
for f in feature_names_org:
    feature_names_rich += [f'{f}', f'{f}_avg', f'{f}_std', f'{f}_q25', f'{f}_q50', f'{f}_q75']


#######################################################################################################################
# Module 4 - Statistics

# Colors used for visualizations in 2_feature_comparison.ipynb notebook as well as in Module 5
PYR_COLOR = (0.416, 0.106, 0.604)
LIGHT_PYR = (0.612, 0.302, 0.8)
PV_COLOR = (0.18, 0.49, 0.196)
LIGHT_PV = (0.376, 0.678, 0.369)
UT_COLOR = (0.082, 0.396, 0.753)
LIGHT_UT = (0.369, 0.573, 0.953)

# Best performing chunk sizes according to the analysis performed in 5_chunking_results.ipynb notebook.
# Later used in Module 5
BEST_WF_CHUNK = 50
BEST_SPATIAL_CHUNK = 25
BEST_ST_CHUNK = 1600


#######################################################################################################################
# Module 5 - Visulaization

# Identifiers of PYR and PV cells used in the figures
PYR_NAME = 'es25nov11_13_3_3'
PV_NAME = 'es25nov11_13_3_11'

# the following values are required to transform the spikes from arbitrary units to Voltage
# this values are based on the ADC configuration
VOL_RANGE = 8
AMPLIFICATION = 1000
NBITS = 16

# Number of spatial events
NUM_EVENTS = 3
