# SpatiotemporalSpiking
This repository contains the code used in Sukman and Stark, 2022, Cortical pyramidal and parvalbumin cells exhibit 
distinct spatiotemporal extracellular electric potentials.

## Requirements
The code was tested on machines running Windows (Windows 10) and Linux (Ubuntu 18), but is expected to work on any OS.

The code was tested with Python 3.9.1 and the packages described in the _requirements.txt_ file. Other versions of
Python and/or the packages are not guaranteed to work or yield similar results. 

Some statistical tests were conducted using MATLAB R2021B. In addition, to calculate mutual information values 
(Module 4), it is required to install the package developed in Timme and Lapish, 2018 (eNeuro).

## Modules
The code should be executed from the _src_ directory and the input to all code segments should be provided by changing
only the _constants.py_ and _paths.py_ files.

The code files, all found under the _src/_ directory, are grouped into __5__ modules:
### 1. Raw data parsing
The module is executed by running _read_data.py_.  
The module is used to read the raw data, and extract only the relevant spikes and assign labels to them (see _Spike 
sorting and ground truth labels_ under _Materials and Methods_). Utility functions for this module are under _data_utils_.
For this module, be sure to update the following (for additional customization, see _constants.py_):

* _constants.py_:
  * SESSION_EMPTY_INDS - A dictionary containing all session names wanted to be read as keys, coupled with a list of 
  shanks to skip. This was manually created based on prior knowledge, yet it can also be extracted automatically from 
  the tagging mat file (see _DATA_MAT_PATH_ below).
  * 
* _paths.py_:
  * DATA_PATH - Path to raw data. Data should contain .spk, .res, and .clu files according to the format presented in 
  Hazan et al., 2006 (J. Neurosci. Methods) and a .stm file with a field of pairs indicating the start and end times of 
  light stimuli (see more information under _Provided data_ below). 
  * DATA_MAT_PATH - Path to mat file containing tagging data. The mat file should have 6 fields, filename (the name of 
  the recording session), shankclu (pairs of recording shank and unit), region (recording region tag, 0 corresponds to 
  nCX; >=1 corresponds to CA1), act (binary list indicating if the unit was 
  activated by light stimuli), exc (binary list indicating if the unit was excitatory), and inh (binary list indicating
  if the unit was inhibitory).
  * DATA_TEMP_PATH - Path in which the module will save the outputs as numpy arrays after removing spikes that occurred 
  during lights stimuli.
  * TEMP_PATH_FULL - Path in which the module will save the outputs as numpy arrays including spikes occurring during 
  light stimuli.
  
### 2. Feature extraction
The module is executed by running _preprocessing_pipeline.py_.  
The module is used to extract features from the outputs of __Module 1__ (See __Tables 1-3__ for the description of the 
features, and subsection _chunking_ of the _Materials and Methods_ for the chunk statistics extracted from the original
features). Utility functions for this module are under _data_utils_ and _features_. For this module, be sure to update
the following (inputs for previous modules should not be changed; for additional customization, see _constants.py_):

* _constants.py_:
  * SEED - The Seed used for randomized chunk partition.
  * CHUNK_SIZES - Chunk sizes to use for chunks, note that 0 means no-chunking. 
  * TRANS_WF - Binary value indicating whether to perform delta-transformation on the waveforms before extracting 
  waveform-based features. If True, only waveform-based features will be extracted.

* _paths.py_:
  * SAVE_PATH - Path to save features and metadata for every unit. This output will not include the chunk statistics. 
  * SAVE_PATH_RICH - Path to save features and metadata for every unit including the chunk statistics for every feature.

### 3. Machine learning
The module is executed by running _ml_pipeline.py_.  
The module is used to create and evaluate cell type classification models based on the outputs of __Module 2__ (See the 
_Classification_ subsection under _Materials and Methods_ for methodology). Outputs of the module include the datasets,
and the evaluation results. Results are saved in 5 files: CSV file including performance and feature importance for
every iteration, modality, and chunk size; NPY file with the raw predictions on the test set for all trained models; 
NPY file with the raw SHAP value for each sample for all trained models; NPY file with the raw predictions on the 
non-trained-upon-region test set if region-based partition was performed; NPY file with the raw SHAP value for each
sample for all trained models on the non-trained-upon-region test set if region-based partition was performed set for
all trained models. Utility functions for this module are under
_ml_. For this module, be sure to update the following (inputs for previous modules should not be changed; for 
additional customization, see _constants.py_):

* _constants.py_:
  * MODALITIES - Names of modalities coupled with feature indices.
  * CHUNKS_MAP - Mapping between modality and chunk sizes to test if wanted to test different chunk sizes for different 
  modalities.
  * RUN_NAME - Identifier of the run, the outputs of the module will be saved in a directory named accordingly.
  * REGION_BASED - Whether to
  * TRAIN_CA1 - Relevant only when REGION_BASED is True, in which case controls the training region.
  * SHUFFLE - Whether to shuffle the labels before training. If 0 or False no shuffling occurs, otherwise repeats 
  shuffle _SHUFFLE_ times. 
  * GS_ZERO - Whether to perform grid search procedure with chunking (True corresponds to no-chunking) 

* _paths.py_:
  * DATASET_PATH - Path in which datasets will be saved.
  * RESULTS_PATH - Path in which results files will be saved.
  * ML_INPUT_PATH - Path to the post-processed data.

### 4. Statistics
The module is executed by running the individual notebooks inside the _statistics_analysis/_ directory.  
The module is used to perform statistical analysis, mainly based on the outputs of __Module 3__. Note that not all tests
are performed in the notebooks. Some statistics are performed in the next module (specifically statistics related to 
__Figure 3C__, __Figure 4__, and __Table 4__). In addition, some statistical tests are performed in MATLAB (specifically 
Kruskal-Wallis tests, correlation significance and mutual information calculation). Note that some results files used
for the analysis were modified to have the importance values of the original features, feature families and spatial
event groups. Code for the manipulation is provided and explained in _statistics_analysis/combine_res.py_. In addition,
p-values based on permutation tests may vary across executions due to the random nature of the sampling. Utility 
functions for this module are under _statistics_. For this module, be sure to update the following (inputs for previous
modules should not be changed):

* _paths.py_:
  * MAIN_RES - Results of the regular execution and feature importance for the original features 
  * BASE_RES - Baseline (using shuffled labels) of the regular execution and feature importance for the original
  features
  * FAMS_RES - Results for the spatial models of the regular execution and feature importance for the spatial feature 
  families 
  * BASE_FAMS_RES - Baseline (using shuffled labels) for the spatial models of the regular execution and feature 
  importance for the spatial feature families 
  * EVENTS_RES - Results for the spatial models of the regular execution and feature importance for the spatial event 
  groups 
  * BASE_EVENTS_RES - Baseline (using shuffled labels) for the spatial models of the regular execution and feature 
  importance for the spatial event groups 
  * REGION_CA1_RES - Results of the CA1-trained models with importance for the original features based on the CA1 test
  set
  * REGION_NCX_RES - Results of the NCX-trained models with importance for the original features based on the NCX test
  set

### 5. Visualization
The module is executed by running _vis_figures.py_.  
The module is used to create the panels from the figures in the manuscript and specific statistical analysis (see above).
Utility functions for this module are under _vis_utils_. Note that some visual aspects of the outputs were added/fixed 
manually. For this module, be sure to update the following (inputs for previous modules should not be changed):

* _paths.py_:
  * MAIN_PREDS - Raw predictions for test set samples for each iteration, modality, and chunk size of the regular 
  execution
  * TRANS_WF_RES - Results of the execution with waveform-based features calculated on the transformed spikes (no chunk
  statistics).
  * CORR_MAT - mat file containing Spearman's correlation coefficients between the features and p-values.
  * MI_MAT - mat file containing mutual information values between the features and p-values.

## Provided data
A dataset is provided at https://doi.org/10.5281/zenodo.7273925 for testing the modules.
* Raw data are provided for a single shank from a single recording session under _data_files/raw_data/es25nov11_13/_. 
Files with .spk, .res, and .clu formats are constructed as described in Hazan et al., 2006. .xml file contains metadata
for the recording session. .stm file contains information about the light stimuli (start and end times, type of stimuli,
intensity of stimuli, and shank).
* Mat file containing tagging information is provided in _data_files/tags.mat_. The file contains information for every
unit from every shank (shankclu) and recording session (filename) about the functionality (excitatory-exc;
inhibitory-inh), responsiveness to light (act) and region (region; 0 corresponds to neocortex and >=1 corresponds to
CA1).
* As the raw data provided for testing the first module are a subset of the full dataset used in the manuscript, all 
post-processed data (after feature extraction) are also provided under _data_files/postprocess_data/_. The directory
includes two sub-directories: _cluster_data/_ containing the features of all three modalities and 
_cluster_data_wf_trans/_ which contains only waveform-based features extracted from the transformed spikes. Note that
_cluster_data/_ does not contain chunk statistics. To extract chunk statistics you may utilize _preprocessing_pipeline.py_.
* Since execution times can be prolonged (see _Replicating results_ below), results files are provided under 
_data_files/results/_ and can be used for Modules 4 and 5. See _paths.py_ for explanations of the different result files
* Statistical data, specifically mat files containing the p-values of the correlation coefficients and mutual 
information values are available under _data_files/statistics/_.

## Replicating results
1) To add chunk statistics to the data, run _preprocessing_pipeline.py_. Be sure to run Module 1 before or 
alternatively manually execute only the function _add_chunk_statistics_. Due to having to load and modify large files,
execution may take a few minutes on a standard pc (tested on 7th generation Intel i7 processor and 16 GB of RAM).
2) Run Module 3 with NUM_ITER=50, and ML_INPUT_PATH set to SAVE_PATH_RICH. This may take around a day on a standard PC, 
to validate the procedure, here and in the next steps, you may run it with fewer iterations. Then repeat with
ML_INPUT_PATH set to SAVE_PATH and adapt MODALITIES in _constants.py_ accordingly (see explanation within the file).
Also remember to change output names to avoid overwriting previously computed performance. Finally, combine the two 
output files created using the function _combine_chunks_ from _statistics_analysis/combine_res.py_ followed by 
_get_feature_imp_, _get_events_imp_ and _get_family_imp_. 
3) To get chance level results, run Module 3 with NUM_ITER=1000, SHUFFLE=1, and ML_INPUT_PATH set to SAVE_PATH_RICH. It
is highly recommended to use CHUNKS_MAP (see _constants.py_) to shorten run times. With a standard PC this process may
take several weeks, mainly due to SHAP analysis when by chance deep trees are trained, hence it is possible to 
change/forgo the grid search procedure though results may vary. The process can be accelerated using parallel 
processing, which allowed us to perform the analysis in less than a week. You should then run _get_feature_imp_ from 
_statistics_analysis/combine_res.py_.
4) Last, execute Module 3 with the REGION_BASED flag set to True to train the models on a single region. Change 
NUM_ITER back to 50 and repeat the process twice with TRAIN_CA1 set to both True and False. You should then run 
_get_feature_imp_ from _statistics_analysis/combine_res.py_.
