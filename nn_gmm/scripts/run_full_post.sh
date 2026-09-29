#!/usr/bin/env zsh

if [ "$#" -ne 2 ]; then
   echo "Usage: $0 <model_dir> <cv_model_dir>"
   exit 1
fi

set -e
model_dir="$1"

imdb_ffp=$wdata/nn_gmm/20251222_CS200_DSSlab_rotd50_imdb.duckdb
echo "imdb_ffp: $imdb_ffp"

### Reference sites hazard
echo "Running reference site DS hazard computations for model dir $model_dir"
python hazard_cmds.py run-site-ds-hazard $model_dir --n-procs 12
python hazard_cmds.py run-site-ds-hazard $model_dir --n-procs 12 --sigma-ept-ffp $model_dir/sigma_ept.csv 0.95
python hazard_cmds.py run-site-ds-hazard $model_dir --n-procs 12 --sigma-ept-ffp $model_dir/sigma_ept.csv 0.05

# Reference sites disaggregation
echo "Running reference site DS disaggregation computations for model dir $model_dir"
python hazard_cmds.py run-site-disagg $model_dir HORC LHUS pSA_0.01 475
python hazard_cmds.py run-site-disagg $model_dir HORC LHUS pSA_0.01 2475
python hazard_cmds.py run-site-disagg $model_dir HORC LHUS pSA_5.0 475
python hazard_cmds.py run-site-disagg $model_dir HORC LHUS pSA_5.0 2475

### Uniform grid
echo "Running uniform grid DS hazard computations for model dir $model_dir"
python hazard_cmds.py compute-uniform-grid-ds-hazard $imdb_ffp $model_dir --n-procs 16



    
