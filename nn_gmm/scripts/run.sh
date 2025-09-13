#!/usr/bin/env zsh

## Test
# python nn_cmds.py train-gmm-cv ./configs/nn_gmm_config_v3.yaml 2 2 --n-epochs 10 --id-suffix cv_v3_10Epochs_2E2SFolds_1000Sites --n-procs 4 --remove-cv-results --n-sites 1000 && \
# python nn_cmds.py train-gmm-cv ./configs/nn_gmm_config_v4.yaml 2 2 --n-epochs 10 --id-suffix cv_v4_10Epochs_2E2SFolds_1000Sites --n-procs 4 --remove-cv-results --n-sites 1000 && \
# python nn_cmds.py train-gmm-cv ./configs/nn_gmm_config_v5.yaml 2 2 --n-epochs 10 --id-suffix cv_v5_10Epochs_2E2SFolds_1000Sites --n-procs 4 --remove-cv-results --n-sites 1000 && \

# python nn_cmds.py train-gmm-cv ./configs/nn_gmm_config_v5.yaml 2 2 --n-epochs 10 --id-suffix cv_v5_10Epochs_2E2Folds_512batchSize_1000Sites --n-procs 4 --n-sites 1000 --remove-cv-results --batch-size 512 && \
# python nn_cmds.py train-gmm-cv ./configs/nn_gmm_config_v1.yaml 3 4 --n-epochs 25 --id-suffix cv_v1_25Epochs_3E4Folds_512batchSize --n-procs 4 --remove-cv-results --batch-size 512 && \
# python nn_cmds.py train-gmm-cv ./configs/nn_gmm_config_v2.yaml 3 4 --n-epochs 25 --id-suffix cv_v2_25Epochs_3E4Folds_512batchSize --n-procs 4 --remove-cv-results --batch-size 512 

# python nn_cmds.py train-gmm-cv ./configs/nn_gmm_config_v3.yaml 3 4 --n-epochs 25 --id-suffix cv_v3_25Epochs_3E4Folds_512batchSize --n-procs 4 --remove-cv-results --batch-size 512 
# python nn_cmds.py run-mera /home/claudy/dev/work/data/nn_gmm/results/0910_1421_cv_v1_25Epochs_3E4Folds_512batchSize --n-procs 4

## Render CV result analysis notebook 
notebook_dir=/home/claudy/dev/work/code/nn_gmm/nn_gmm/result_notebooks
cur_dir=/home/claudy/dev/work/data/nn_gmm/results/0912_1844_cv_v3_25Epochs_3E4Folds_512batchSize
quarto render $notebook_dir/cv_result_analysis.ipynb --execute --to html  -P result_dir:$cur_dir && \
mv $notebook_dir/cv_result_analysis.html $cur_dir/cv_result_analysis.html

python nn_cmds.py run-mera /home/claudy/dev/work/data/nn_gmm/results/0911_0917_cv_v2_25Epochs_3E4Folds_512batchSize --n-procs 4 2>&1 | tee run_mera_v2_25Epochs.log
python nn_cmds.py run-mera /home/claudy/dev/work/data/nn_gmm/results/0912_1844_cv_v3_25Epochs_3E4Folds_512batchSize --n-procs 4 2>&1 | tee run_mera_v3_25Epochs.log


# Basin & site map
# python plot_cmds.py basin-site-map /Users/claudy/dev/work/data/nn_gmm/20250606_CS200m_imdb.db /Users/claudy/dev/work/tmp/spatial/basin_sites_level_REAL.png --site-level -1 --basin-dir /Users/claudy/dev/work/code/nn_gmm/nn_gmm/resources/basin_boundaries &
# python plot_cmds.py basin-site-map /Users/claudy/dev/work/data/nn_gmm/20250606_CS200m_imdb.db /Users/claudy/dev/work/tmp/spatial/basin_sites_level_0.png --site-level 0 --basin-dir /Users/claudy/dev/work/code/nn_gmm/nn_gmm/resources/basin_boundaries &
# wait

# python plot_cmds.py basin-site-map /Users/claudy/dev/work/data/nn_gmm/20250606_CS200m_imdb.db /Users/claudy/dev/work/tmp/spatial/basin_sites_level_1.png --site-level 1 --basin-dir /Users/claudy/dev/work/code/nn_gmm/nn_gmm/resources/basin_boundaries &
# python plot_cmds.py basin-site-map /Users/claudy/dev/work/data/nn_gmm/20250606_CS200m_imdb.db /Users/claudy/dev/work/tmp/spatial/basin_sites_level_2.png --site-level 2 --basin-dir /Users/claudy/dev/work/code/nn_gmm/nn_gmm/resources/basin_boundaries &
# wait 

# python plot_cmds.py basin-site-map /Users/claudy/dev/work/data/nn_gmm/20250606_CS200m_imdb.db /Users/claudy/dev/work/tmp/spatial/basin_sites_level_3.png --site-level 3 --basin-dir /Users/claudy/dev/work/code/nn_gmm/nn_gmm/resources/basin_boundaries &


