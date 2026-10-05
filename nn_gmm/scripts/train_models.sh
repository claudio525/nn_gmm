#!/usr/bin/env zsh
set -e

SCRIPT_DIR="${0:a:h}"


out_dir=${wdata}/nn_gmm/results
echo "out_dir: $out_dir"


# -------------------------------------- Base Model ----------------------------------------
 
## CV
python nn_cmds.py train-gmm-cv ./configs/nn_gmm_configs/base_gmm_config_v6.yaml 6 6 --n-epochs 25 --id-suffix cv_baseV6_25Epochs_6E6Folds --n-procs 9 --remove-cv-results 

## Full
python nn_cmds.py train-full-gmm ./configs/nn_gmm_configs/base_gmm_config_v6.yaml --n-epochs 10 --id-suffix full_baseV6_10Epochs


# -------------------------- Location-Specific Model ---------------------------------------

### Location-Specific Model
cv_rel_base_model_dir=nn_gmm/results/0922_1644_cv_baseV6_25Epochs_6E6Folds
full_rel_base_model_dir=nn_gmm/results/0922_1959_full_baseV6_10Epochs

## CV
python nn_cmds.py train-loc-adj-cv ./configs/nn_gmm_configs/loc_adj_config_v4.yaml --n-epochs 25 --id-suffix cv_baseV6_locAdjV4_25Epochs_6E6Folds --rel-base-model-dir $cv_rel_base_model_dir --n-procs 9 --remove-cv-results

## Full
python nn_cmds.py train-full-loc-adj-model ./configs/nn_gmm_configs/loc_adj_config_v4.yaml --n-epochs 15 --id-suffix full_baseV6_locAdjV4_10Epochs --rel-base-model-dir $full_rel_base_model_dir 


