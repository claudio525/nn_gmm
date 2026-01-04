#!/usr/bin/env zsh
# Generates plots for a CV NN-GMM model.

if [ -z "$1" ]; then
    echo "Error: model_dir argument is required"
    echo "Usage: $0 <model_dir>"
    exit 1
fi

model_dir=$1


out_dir=$model_dir/plots
mkdir -p $out_dir

# Generate spatial site bias/residual std plots
python plot_cmds.py nn-site-bias-res-std $model_dir pSA_1.0 pSA_3.0 pSA_5.0 pSA_10.0 $out_dir --n-procs 4 --grid-spacing 100e/100e

# Generate spatial site-term maps
if [ -d "$model_dir/mera_site_term" ]; then
    python plot_cmds.py nn-site-term-map $model_dir pSA_1.0 pSA_3.0 pSA_5.0 pSA_10.0 $out_dir --n-procs 4 --grid-spacing 100e/100e
fi

# Generate spatial site folds map
# python plot_cmds.py site-folds-map $model_dir $out_dir/site_folds_map.png
