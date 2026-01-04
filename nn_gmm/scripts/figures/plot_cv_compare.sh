#!/usr/bin/env zsh

if [ -z "$1" ] || [ -z "$2" ] || [ -z "$3" ]; then
    echo "Error: All arguments are required"
    echo "Usage: $0 <model_dir1> <model_dir2> <output_dir>"
    exit 1
fi

model_dir1=$1
model_dir2=$2
out_dir=$3

mkdir -p $out_dir

# Site bias histogram
python plot_cmds.py nn-gmm-compare-site-bias-histogram $model_dir1 $model_dir2 $out_dir

# Site bias & residual standard deviation comparison (wrt. pSA period)
python plot_cmds.py nn-gmm-compare-site-bias-res-std $model_dir1 $model_dir2 $out_dir 

# Site term comparison
python gen_paper_figures.py mera-model-bias-std $model_dir1 $model_dir2 $out_dir

# Basin site term comparison (wrt. pSA period)
python plot_cmds.py mera-basin-site-term $model_dir1 $model_dir2 $out_dir

