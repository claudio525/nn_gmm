#!/usr/bin/env zsh
# Generates the paper figures

default_fig_size="6.5,4"
export fig_size=$default_fig_size
export fig_format="png"
export fig_dpi="900"
export fig_font_size="8"
export fig_linewidth="2.0"
export fig_group_linewidth="1.0"

SCRIPT_DIR="${0:a:h}"

imdb_ffp=$wdata/nn_gmm/20251222_CS200_DSSlab_rotd50_imdb.duckdb
echo "imdb_ffp: $imdb_ffp"
emp_ds_hazard_results=$wdata/nn_gmm/results/emp_ds_hazard
echo "emp_ds_hazard_results: $emp_ds_hazard_results"
cv_base_model_dir=$wdata/nn_gmm/results/1231_1247_cv_v1_25Epochs_6E6Folds
echo "cv_base_model_dir: $cv_base_model_dir"
full_base_model_dir=$wdata/nn_gmm/results/1230_1645_full_v1_3Epochs
echo "full_base_model_dir: $full_base_model_dir"
cv_location_model_dir=$wdata/nn_gmm/results/0101_2126_cv_locAdjV1_25Epochs_6E6Folds
echo "cv_location_model_dir: $cv_location_model_dir"
full_location_model_dir=$wdata/nn_gmm/results/0105_1316_full_locAdjV1_5Epochs
echo "full_location_model_dir: $full_location_model_dir"
out_dir=~/dev/tmp_share/nn_gmm/paper_figures
echo "out_dir: $out_dir"

# ### Mixed Effects Regression Results
# echo "Generating mixed effects regression results..."
# python gen_paper_figures.py mera-site-term $cv_base_model_dir $cv_location_model_dir $out_dir

# ### Basin Site Term Comparison
# echo "Generating basin site term comparison..."
# python gen_paper_figures.py mera-basin-site-term $cv_base_model_dir $cv_location_model_dir $out_dir

# ### Site Term Maps
# echo "Generating site term maps..."
# python gen_paper_figures.py nn-gmm-site-term-map $cv_base_model_dir $out_dir pSA_5.0 base --grid-spacing 50e/50e
# python gen_paper_figures.py nn-gmm-site-term-map $cv_location_model_dir $out_dir pSA_5.0 loc --grid-spacing 50e/50e

# ### Model Trends
# echo "Generating model trend figures..."
# export fig_size="3.25,3"
# python gen_paper_figures.py model-trends $cv_base_model_dir $cv_location_model_dir $SCRIPT_DIR/../configs/figure_configs/mag_moderate_dist_model_trend.yaml $out_dir pSA_0.5 pSA_1.0 pSA_5.0 pSA_10.0 --locations HORC --locations LHUS
# export fig_size=$default_fig_size

# ### Site hazard plots
# echo "Generating site hazard plots..."
# export fig_size="3.25,3"
# python gen_paper_figures.py site-hazard-plot $full_base_model_dir/ds_hazard/ref_sites $full_location_model_dir/ds_hazard/ref_sites $emp_ds_hazard_results/ref_sites $out_dir
# export fig_size=$default_fig_size

# ### Site UHS plots
# echo "Generating site UHS plots..."
# export fig_size="3.25,3"
# python gen_paper_figures.py site-uhs-plot $full_base_model_dir/ds_hazard/ref_sites $full_location_model_dir/ds_hazard/ref_sites $emp_ds_hazard_results/ref_sites $out_dir
# export fig_size=$default_fig_size

### DS hazard maps
# echo "Generating DS hazard maps for pSA(5.0s) and RP=2475"
# python gen_paper_figures.py ds-hazard-map $imdb_ffp $full_base_model_dir/ds_hazard/uniform_grid $out_dir pSA_5.0 2475 "Base Model - DS" ds_base 100e/100e 
# python gen_paper_figures.py ds-hazard-map $imdb_ffp $full_location_model_dir/ds_hazard/uniform_grid $out_dir pSA_5.0 2475 "Location Model - DS" ds_loc 100e/100e 
# python gen_paper_figures.py ds-hazard-map $imdb_ffp $emp_ds_hazard_results/uniform_grid $out_dir pSA_5.0 2475 "Empirical GMM Logic Tree - DS" ds_emp 100e/100e 

# echo "Generating DS hazard maps for pSA(0.5s) and RP=2475"
# python gen_paper_figures.py hazard-map $imdb_ffp $full_base_model_dir/ds_hazard/uniform_grid $out_dir pSA_0.5 2475 "Base Model - DS" ds_base 100e/100e 
# python gen_paper_figures.py ds-hazard-map $imdb_ffp $full_location_model_dir/ds_hazard/uniform_grid $out_dir pSA_0.5 2475 "Location Model - DS" ds_loc 100e/100e 
# python gen_paper_figures.py ds-hazard-map $imdb_ffp $emp_ds_hazard_results/uniform_grid $out_dir pSA_0.5 2475 "Empirical GMM Logic Tree - DS" ds_emp 100e/100e 

# echo "Generating DS hazard maps for pSA(5.0s) and RP=475"
# # python gen_paper_figures.py ds-hazard-map $imdb_ffp $full_base_model_dir/ds_hazard/uniform_grid $out_dir pSA_5.0 475 "Base Model - DS" ds_base 100e/100e 
# python gen_paper_figures.py ds-hazard-map $imdb_ffp $full_location_model_dir/ds_hazard/uniform_grid $out_dir pSA_5.0 475 "Location Model - DS" ds_loc 100e/100e 
# # python gen_paper_figures.py ds-hazard-map $imdb_ffp $emp_ds_hazard_results/uniform_grid $out_dir pSA_5.0 475 "Empirical GMM Logic Tree - DS" ds_emp 100e/100e 

# echo "Generating DS hazard maps for pSA(0.5s) and RP=475"
# # python gen_paper_figures.py ds-hazard-map $imdb_ffp $full_base_model_dir/ds_hazard/uniform_grid $out_dir pSA_0.5 475 "Base Model - DS" ds_base 100e/100e 
# python gen_paper_figures.py ds-hazard-map $imdb_ffp $full_location_model_dir/ds_hazard/uniform_grid $out_dir pSA_0.5 475 "Location Model - DS" ds_loc 100e/100e 
# # python gen_paper_figures.py ds-hazard-map $imdb_ffp $emp_ds_hazard_results/uniform_grid $out_dir pSA_0.5 475 "Empirical GMM Logic Tree - DS" ds_emp 100e/100e 


### DS Hazard ratio maps
# echo "Generating DS hazard ratio maps for ln(emp/base)"
# python gen_paper_figures.py hazard-ratio-map $emp_ds_hazard_results/uniform_grid $full_base_model_dir/ds_hazard/uniform_grid $imdb_ffp $out_dir pSA_5.0 2475 "ln(Empirical/Base Model)" ratio_emp_base 100e/100e  --cb-max 1.0
# python gen_paper_figures.py hazard-ratio-map $emp_ds_hazard_results/uniform_grid $full_base_model_dir/ds_hazard/uniform_grid $imdb_ffp $out_dir pSA_0.5 2475 "ln(Empirical/Base Model)" ratio_emp_base 100e/100e  --cb-max 1.0

# echo "Generating DS hazard ratio maps for ln(base/loc)"
# python gen_paper_figures.py hazard-ratio-map $full_base_model_dir/ds_hazard/uniform_grid $full_location_model_dir/ds_hazard/uniform_grid $imdb_ffp $out_dir pSA_5.0 2475 "ln(Base Model/Location Model)" ratio_base_loc 100e/100e --cb-max 1.0
# python gen_paper_figures.py hazard-ratio-map $full_base_model_dir/ds_hazard/uniform_grid $full_location_model_dir/ds_hazard/uniform_grid $imdb_ffp $out_dir pSA_0.5 2475 "ln(Base Model/Location Model)" ratio_base_loc 100e/100e --cb-max 1.0

# ### Total hazard maps (CS fault + DS)
# # echo "Generating total hazard maps for pSA(5.0s) and RP=2475"
# python gen_paper_figures.py total-hazard-map $imdb_ffp $full_base_model_dir/ds_hazard/uniform_grid $out_dir pSA_5.0 2475 "CS 25.6 + Base Model" total_base 50e/50e  
# python gen_paper_figures.py total-hazard-map $imdb_ffp $full_location_model_dir/ds_hazard/uniform_grid $out_dir pSA_5.0 2475 "CS 25.6 + Location Model" total_loc 50e/50e 
# python gen_paper_figures.py total-hazard-map $imdb_ffp $emp_ds_hazard_results/uniform_grid $out_dir pSA_5.0 2475 "CS 25.6 + Empirical Logic Tree" total_emp 50e/50e  

# # echo "Generating total hazard maps for pSA(0.5s) and RP=2475"
# python gen_paper_figures.py total-hazard-map $imdb_ffp $full_base_model_dir/ds_hazard/uniform_grid $out_dir pSA_0.5 2475 "CS 25.6 + Base Model" total_base 50e/50e  
# python gen_paper_figures.py total-hazard-map $imdb_ffp $full_location_model_dir/ds_hazard/uniform_grid $out_dir pSA_0.5 2475 "CS 25.6 + Location Model" total_loc 50e/50e  
# python gen_paper_figures.py total-hazard-map $imdb_ffp $emp_ds_hazard_results/uniform_grid $out_dir pSA_0.5 2475 "CS 25.6 + Empirical Logic Tree" total_emp 50e/50e  



# --------------------------------------- Electronic Supplementary Material Figures ---------------------------------------

# ### DS Hazard ratio maps
echo "Generating DS hazard ratio maps for ln(emp/base) at RP=475"
python gen_paper_figures.py hazard-ratio-map $emp_ds_hazard_results/uniform_grid $full_base_model_dir/ds_hazard/uniform_grid $imdb_ffp $out_dir pSA_5.0 475 "ln(Empirical/Base Model)" ratio_emp_base 100e/100e  --cb-max 1.0
python gen_paper_figures.py hazard-ratio-map $emp_ds_hazard_results/uniform_grid $full_base_model_dir/ds_hazard/uniform_grid $imdb_ffp $out_dir pSA_0.5 475 "ln(Empirical/Base Model)" ratio_emp_base 100e/100e  --cb-max 1.0

echo "Generating DS hazard ratio maps for ln(base/loc) at RP=475"
python gen_paper_figures.py hazard-ratio-map $full_base_model_dir/ds_hazard/uniform_grid $full_location_model_dir/ds_hazard/uniform_grid $imdb_ffp $out_dir pSA_5.0 475 "ln(Base Model/Location Model)" ratio_emp_loc 100e/100e  --cb-max 1.0
python gen_paper_figures.py hazard-ratio-map $full_base_model_dir/ds_hazard/uniform_grid $full_location_model_dir/ds_hazard/uniform_grid $imdb_ffp $out_dir pSA_0.5 475 "ln(Base Model/Location Model)" ratio_emp_loc 100e/100e  --cb-max 1.0