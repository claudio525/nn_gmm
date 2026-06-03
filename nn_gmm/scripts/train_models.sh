#!/usr/bin/env zsh
set -e

SCRIPT_DIR="${0:a:h}"


out_dir=${wdata}/nn_gmm/results
echo "out_dir: $out_dir"


# -------------------------------------- Base Model ----------------------------------------
 
## CV
# python nn_cmds.py train-gmm-cv ./configs/nn_gmm_configs/base_gmm_config_v3.yaml 6 6 --n-epochs 25 --id-suffix cv_baseV3_25Epochs_6E6Folds_final --n-procs 9 --remove-cv-results 

## Full
# python nn_cmds.py train-full-gmm ./configs/nn_gmm_configs/base_gmm_config_v3.yaml --n-epochs 3 --id-suffix full_baseV3_3Epochs_final 
# python nn_cmds.py train-full-gmm ./configs/nn_gmm_configs/base_gmm_config_v3.yaml --n-epochs 4 --id-suffix full_baseV3_4Epochs_final 
# python nn_cmds.py train-full-gmm ./configs/nn_gmm_configs/base_gmm_config_v3.yaml --n-epochs 5 --id-suffix full_baseV3_5Epochs_final 
# python nn_cmds.py train-full-gmm ./configs/nn_gmm_configs/base_gmm_config_v3.yaml --n-epochs 7 --id-suffix full_baseV3_7Epochs_final 
# python nn_cmds.py train-full-gmm ./configs/nn_gmm_configs/base_gmm_config_v3.yaml --n-epochs 13 --id-suffix full_baseV3_13Epochs_final 

# -------------------------- Location-Specific Model ---------------------------------------

### Location-Specific Model
# cv_rel_base_model_dir=nn_gmm/results/0528_0949_cv_baseV3_25Epochs_6E6Folds_final
# full_rel_base_model_dir=nn_gmm/results/0528_1420_full_baseV3_3Epochs_final
full_rel_base_model_dir=nn_gmm/results/0603_1224_full_baseV3_4Epochs_final

## CV
# python nn_cmds.py train-loc-adj-cv ./configs/nn_gmm_configs/loc_adj_config_v1.yaml --n-epochs 25 --id-suffix cv_locAdjV1_25Epochs_6E6Folds_final --rel-base-model-dir $cv_rel_base_model_dir --n-procs 9 

## Full
python nn_cmds.py train-full-loc-adj-model ./configs/nn_gmm_configs/loc_adj_config_v1.yaml --n-epochs 10 --id-suffix full_locAdjV1_10Epochs_final --rel-base-model-dir $full_rel_base_model_dir 


