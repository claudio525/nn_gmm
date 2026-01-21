#!/usr/bin/env zsh
# Generates plots for a full NN-GMM model.

if [ -z "$1" ]; then
    echo "Error: model_dir argument is required"
    echo "Usage: $0 <model_dir>"
    exit 1
fi

model_dir=$1

imdb_ffp=$wdata/nn_gmm/20251222_CS200_DSSlab_rotd50_imdb.duckdb
emp_ds_hazard_dir=$wdata/nn_gmm/results/emp_ds_hazard

out_dir=$model_dir/plots
mkdir -p $out_dir

# # Generate hazard & UHS plots for reference sites
# python plot_cmds.py gen-site-hazard-plots $model_dir/ds_hazard/ref_sites $out_dir --emp-ds-results-dir $emp_ds_hazard_dir/ref_sites
# python plot_cmds.py gen-site-uhs-plots $model_dir/ds_hazard/ref_sites $out_dir --emp-ds-results-dir $emp_ds_hazard_dir/ref_sites

# # Generate DS hazard maps
python plot_cmds.py hazard-map $wdata/nn_gmm/20251222_CS200_DSSlab_rotd50_imdb.duckdb $model_dir/ds_hazard/uniform_grid $out_dir --ims pSA_0.01 --ims pSA_0.1 --ims pSA_0.5 --ims pSA_1.0  --ims pSA_3.0 --ims pSA_5.0 --ims pSA_10.0 --rps 475 --rps 975 --rps 2475 --n-procs 10

# # Generate DS hazard ratio maps (wrt. empirical GMMs)
# python plot_cmds.py ds-hazard-ratio-map $imdb_ffp $emp_ds_hazard_dir/uniform_grid $model_dir/ds_hazard/uniform_grid $out_dir "ln(Empirical/Base Model)" ratio_emp_base --ims pSA_0.01 --ims pSA_0.1 --ims pSA_0.5 --ims pSA_1.0  --ims pSA_3.0 --ims pSA_5.0 --ims pSA_10.0 --rps 475 --rps 975 --rps 2475 --n-procs 10

# Generate total hazard maps (CS fault + DS)
# python plot_cmds.py hazard-map $imdb_ffp $model_dir/ds_hazard/uniform_grid $out_dir --ims pSA_0.01 --ims pSA_0.1 --ims pSA_0.5 --ims pSA_1.0  --ims pSA_3.0 --ims pSA_5.0 --ims pSA_10.0 --rps 475 --rps 975 --rps 2475 --n-procs 10 --add-flt-hazard
