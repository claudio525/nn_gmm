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
# python plot_cmds.py nn-site-bias-res-std $model_dir pSA_1.0 pSA_3.0 pSA_5.0 pSA_10.0 $out_dir --n-procs 4 --grid-spacing 250e/250e

# Generate spatial site-term and remaining residual maps
if [ -d "$model_dir/mera_site_term" ]; then
    python plot_cmds.py nn-site-term-map $model_dir pSA_1.0 pSA_3.0 pSA_5.0 pSA_10.0 $out_dir --n-procs 4 --grid-spacing 250e/250e
    # python plot_cmds.py nn-rem-residual-map $model_dir pSA_1.0 pSA_3.0 pSA_5.0 pSA_10.0 $out_dir --n-procs 4 --grid-spacing 250e/250e
fi

# Generate spatial mean predicted std plots
python plot_cmds.py cv-mean-pred-std-map $model_dir $out_dir pSA_1.0 pSA_3.0 pSA_5.0 pSA_10.0 --n-procs 4 --grid-spacing 250e/250e


