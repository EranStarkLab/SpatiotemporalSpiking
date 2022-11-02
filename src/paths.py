MAIN_PATH = 'path/to/data/directory/'

#######################################################################################################################
# Paths For inputs and outputs of Modules 1-3
# Paths to raw data used for feature extraction
DATA_PATH = MAIN_PATH + 'raw_data/'
DATA_MAT_PATH = MAIN_PATH + 'tags.mat'

# Paths to Outputs of Module 1
DATA_TEMP_PATH = MAIN_PATH + 'temp_state/'
TEMP_PATH_FULL = MAIN_PATH + 'temp_state_full/'

# Paths to outputs of Module 2
SAVE_PATH = MAIN_PATH + 'postprocess_data/clusters_data/'
SAVE_PATH_RICH = MAIN_PATH + 'postprocess_data/clusters_data_rich/'  # notice that this data is not provided
SAVE_PATH_WF_TRANS = MAIN_PATH + 'postprocess_data/clusters_data_wf_trans/'

# Paths to inputs of Module 3. This should be one of the outputs of module 2
ML_INPUT_PATH = SAVE_PATH_RICH

# Paths to outputs of Module 3
DATASET_PATH = MAIN_PATH + 'datasets/'
RESULTS_PATH = MAIN_PATH + 'results/'

# Paths to inputs for Modules 4 and 5
STATS_PATH = MAIN_PATH + 'statistics/'

# Paths to outputs of Module 5
FIG_PATH = MAIN_PATH + 'figures/'

#######################################################################################################################
# Paths to results and analysis files
# Note that these csv files are different than the outputs of Module 3 as the importance values were grouped

# Results for every modality (waveform-based, spike-timing, and spatial) without and with chunking to 1,600-25-
# chunk-spikes ran for 50 iterations. Chunk size of zero represents no chunking. Importance values are grouped based on
# the original features.
MAIN_RES = RESULTS_PATH + 'results_rf_combined.csv'

# Raw predictions of the results. Note that those are raw predictions of the models described above, only without the
# chunk statistics, hence, chunking performance will not be similar. Order in file is the same as in the previous csv.
MAIN_PREDS = RESULTS_PATH + 'preds_rf.npy'

# Results for Baseline with 1,000 iterations each with shuffled labels. For every modality, only the best chunk size
# according to the previous execution was tested. results for chunk size of 0 represent the no-chunking equivalent used
# for the grid search procedure.
BASE_RES = RESULTS_PATH + 'results_shuffles_combined.csv'

# Results for the spatial modality with importance values are grouped based on the spatial feature families.
FAMS_RES = RESULTS_PATH + 'results_rf_families.csv'

# Baseline results for the spatial modality with importance values are grouped based on the spatial feature families.
BASE_FAMS_RES = RESULTS_PATH + 'results_shuffles_families.csv'

# Results for the spatial modality with importance values are grouped based on the spatial events.
EVENTS_RES = RESULTS_PATH + 'results_rf_events.csv'

# Baseline results for the spatial modality with importance values are grouped based on the spatial events.
BASE_EVENTS_RES = RESULTS_PATH + 'results_shuffles_events.csv'

# No chunking results for waveform-based features extracted from the transformed spikes.
TRANS_WF_RES = RESULTS_PATH + 'results_rf_trans_wf.csv'

# Raw predictions of the classification based on waveform features extracted from the transformed spikes.
TRANS_WF_PREDS = RESULTS_PATH + 'preds_rf_trans_wf.npy'

# Results for CA1 trained models tested on CA1 and neocortical data based on 50 iterations. Importance was calculated
# based on the training-region test set and values are grouped by the original features.
REGION_CA1_RES = RESULTS_PATH + 'results_rf_ca1_train_ca1_imp.csv'

# Results for nCX trained models tested on CA1 and neocortical data based on 50 iterations. Importance was calculated
# based on the training-region test set and values are grouped by the original features.
REGION_NCX_RES = RESULTS_PATH + 'results_rf_ncx_train_ncx_imp.csv'

#######################################################################################################################
# Extracted statistics

# P-values for the correlation coefficients
CORR_MAT = STATS_PATH + 'spearman.mat'

# MI values and p-values
MI_MAT = STATS_PATH + 'MIs.mat'
