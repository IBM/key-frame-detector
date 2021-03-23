## Scripts 
Duke_keyframe_crossval_sets.xlsx: This excelsheet contains the manual annotations for each B-scan in the Duke dataset. It also contains the 10-fold crossvalidation split. 

train_denovo_keyframe_detector: This script trains the denovo network using a 10-fold cross validation experiment. 
The xlsx sheet Duke_keyframe_crossval_sets.xlsx contain the split of the volumes/Bscans into 10-sets of 75%, 10%, 15% splits into training/validation/testing sets. 

train_TL_keyframe_detector: Does the same for the TL network.

visualize_training_log: Visualize the training/validation losses/metrics and generate the figures shown in the paper. 
