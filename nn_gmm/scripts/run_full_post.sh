#!/usr/bin/env zsh

if [ "$#" -ne 1 ]; then
   echo "Usage: $0 <model_dir>"
   exit 1
fi

set -e
model_dir="$1"

imdb_ffp=$wdata/nn_gmm/20251222_CS200_DSSlab_rotd50_imdb.duckdb
echo "imdb_ffp: $imdb_ffp"

### Reference sites
echo "Running reference site DS hazard computations for model dir $model_dir"
python hazard_cmds.py run-site-ds-hazard $model_dir --n-procs 12

### Uniform grid
echo "Running uniform grid DS hazard computations for model dir $model_dir"
python hazard_cmds.py compute-uniform-grid-ds-hazard $imdb_ffp $model_dir --n-procs 16



    
