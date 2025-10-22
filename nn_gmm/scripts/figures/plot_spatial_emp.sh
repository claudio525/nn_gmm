#!/usr/bin/env zsh

if [ -z "$1" ]; then
    echo "Error: model_dir argument is required"
    echo "Usage: $0 <model_dir>"
    exit 1
fi

model_dir=$1

out_dir=$model_dir/plots/emp_gmm
mkdir -p $out_dir
python plot_cmds.py emp-gmm-bias-res-std $model_dir $wdata/nn_gmm/20250616_CS200m_empdb.db pSA_0.01 pSA_0.1 pSA_0.5 pSA_1.0 pSA_3.0 pSA_5.0 pSA_10.0 $out_dir --n-procs 4 --grid-spacing 100e/100e