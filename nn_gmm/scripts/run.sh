#!/usr/bin/env zsh

function log_gpu_mem {
    echo "timestamp, used_memory, utilization" > $1
    while true; do
        echo "$(date -Iseconds),$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits)" >> $1
        sleep 1
    done
}

function csnotify {
   curl -d $1 ntfy.sh/W7T2QKNDH9Z4E3VJPRY8XACUL
}


log_gpu_mem gpu_memory_usage.log &
bg_pid=$!

# python nn_cmds.py train-gmm-cv ./configs/nn_gmm_configs/base_gmm_config_v1.yaml 6 6 --n-epochs 25 --id-suffix cv_base_v1_25Epochs_6E6Folds --n-procs 9 --remove-cv-results && csnotify "GMM CV training complete." || csnotify "GMM CV training failed."

# python nn_cmds.py train-gmm-cv ./configs/nn_gmm_configs/base_gmm_config_v1.yaml 6 6 --n-epochs 500 --id-suffix cv_v1_500Epochs_6E6Folds --n-procs 9 && csnotify "GMM CV training complete." || csnotify "GMM CV training failed."
# python nn_cmds.py train-loc-adj-cv ./configs/nn_gmm_configs/loc_adj_config_v1.yaml --n-epochs 25 --id-suffix cv_locAdjV1_25Epochs_6E6Folds --rel-base-model-dir nn_gmm/results/1231_1247_cv_v1_25Epochs_6E6Folds --n-procs 9 && csnotify "GMM CV training complete." || csnotify "GMM CV training failed. "

# python nn_cmds.py train-gmm-cv ./configs/nn_gmm_configs/base_gmm_config_trial_095.yaml 6 6 --n-epochs 25 --id-suffix cv_trial095_25Epochs_6E6Folds --n-procs 9 && csnotify "GMM CV training complete." || csnotify "GMM CV training failed."

# python nn_cmds.py run-mera /home/claudy/dev/work/data/nn_gmm/results/0225_0948_cv_base_v1_25Epochs_6E6Folds --site-term --n-procs 3 && csnotify "MERA run complete" || csnotify "MERA run failed"

# python nn_cmds.py train-full-gmm ./configs/nn_gmm_configs/base_gmm_config_v1.yaml --n-epochs 3 --id-suffix full_v1_3Epochs && csnotify "Full GMM model training complete." || csnotify "Full GMM model training failed."
# python nn_cmds.py train-full-loc-adj-model ./configs/nn_gmm_configs/loc_adj_config_v1.yaml --n-epochs 5 --id-suffix full_locAdjV1_5Epochs --rel-base-model-dir nn_gmm/results/1230_1645_full_v1_3Epochs && csnotify "Full loc-adj model training complete." || csnotify "Full loc-adj model training failed."

# python hazard_cmds.py compute-uniform-grid-ds-hazard /home/claudy/dev/work/data/nn_gmm/20251222_CS200_DSSlab_rotd50_imdb.duckdb /home/claudy/dev/work/data/nn_gmm/results/0105_1316_full_locAdjV1_5Epochs --n-procs 12 && csnotify "NN grid hazard complete" || csnotify "NN grid hazard failed"

kill $bg_pid
wait $bg_pid 2>/dev/null



# python nn_cmds.py train-full-gmm ./configs/nn_gmm_configs/nn_gmm_config_v3_locSiteCond.yaml --n-epochs 20 --id-suffix full_v3_locSiteCond_20Epochs 


# dirs=(
#     /home/claudy/dev/work/data/nn_gmm/results/1101_1435_cv_v1_25Epochs_6E6Folds
# )

# ## Render CV outlier
# notebook_dir=/home/claudy/dev/work/code/nn_gmm/nn_gmm/explore

# for cur_dir in "${dirs[@]}"; do
#     echo "Processing: $cur_dir"
#     quarto render $notebook_dir/cv_outlier_analysis.ipynb --execute --to html -P results_dir:$cur_dir && \
#     mv $notebook_dir/cv_outlier_analysis.html $cur_dir/cv_outlier_analysis.html
# done

# csnotify "CV outlier analysis notebooks rendering complete."


# Render CV result analysis notebook 
# notebook_dir=/home/claudy/dev/work/code/nn_gmm/nn_gmm/result_notebooks
# cur_dir=/home/claudy/dev/work/data/nn_gmm/results/1107_1533_cv_v1_25Epochs_6E6Folds_lrReduc
# quarto render $notebook_dir/cv_result_analysis.ipynb --execute --to html  -P result_dir:$cur_dir && \
# mv $notebook_dir/cv_result_analysis.html $cur_dir/cv_result_analysis.html ; csnotify "CV result analysis notebook rendering complete."

# Render exploratory notebook for outlier analysis
# notebook_dir=/home/claudy/dev/work/code/nn_gmm/nn_gmm/result_notebooks
# cur_dir=/home/claudy/dev/work/data/nn_gmm/results/1107_1954_cv_v1_25Epochs_6E6Folds_HP1104-1103_trial57_lrReduc
# quarto render $notebook_dir/cv_result_analysis.ipynb --execute --to html  -P result_dir:$cur_dir && \
# mv $notebook_dir/cv_result_analysis.html $cur_dir/cv_result_analysis.html ; csnotify "CV result analysis notebook rendering complete."


# notebook_dir=/home/claudy/dev/work/code/nn_gmm/nn_gmm/explore
# cur_dir=/home/claudy/dev/work/data/nn_gmm/results/0930_1813_cv_v3_locSiteCond_25Epochs_3E4Folds_seed72
# quarto render $notebook_dir/cv_outlier_analysis.ipynb --execute --to html  -P results_dir:$cur_dir && \
# mv $notebook_dir/cv_outlier_analysis.html $cur_dir/cv_outlier_analysis.html


