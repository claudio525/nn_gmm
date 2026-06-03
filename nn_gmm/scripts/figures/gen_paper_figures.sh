#!/usr/bin/env zsh
# Generates the paper figures

default_fig_size="6.5,4"
export fig_size=$default_fig_size
export fig_format="png"
# export fig_dpi="900"
export fig_dpi="500"
export fig_font_size="8"
export fig_linewidth="2.0"
export fig_group_linewidth="1.25"
export fig_minor_linewidth="0.75"

SCRIPT_DIR="${0:a:h}"

imdb_ffp=$wdata/nn_gmm/20251222_CS200_DSSlab_rotd50_imdb.duckdb
echo "imdb_ffp: $imdb_ffp"

test_events_ffp=$wdata/nn_gmm/test_events.npy
echo "test_events_ffp: $test_events_ffp"

emp_ds_hazard_results=$wdata/nn_gmm/results/emp_ds_hazard
echo "emp_ds_hazard_results: $emp_ds_hazard_results"

cv_base_model_dir=$wdata/nn_gmm/results/0528_0949_cv_baseV3_25Epochs_6E6Folds_final
echo "cv_base_model_dir: $cv_base_model_dir"

full_base_model_dir=$wdata/nn_gmm/results/0603_1010_full_baseV3_3Epochs_final
echo "full_base_model_dir: $full_base_model_dir"

cv_location_model_dir=$wdata/nn_gmm/results/0528_1547_cv_locAdjV1_25Epochs_6E6Folds_final
echo "cv_location_model_dir: $cv_location_model_dir"

full_location_model_dir=$wdata/nn_gmm/results/0603_1102_full_locAdjV1_3Epochs_final
echo "full_location_model_dir: $full_location_model_dir"

# out_dir=~/dev/tmp_share/nn_gmm/paper_figures
out_dir=~/dev/tmp_share/nn_gmm/20260602_paper_figures
# out_dir=~/dev/tmp_share/nn_gmm/test
echo "out_dir: $out_dir"

### Fault Map
python gen_paper_figures.py fault-map $out_dir $imdb_ffp

### Magnitude-Rrrup Scatter Plot
python gen_paper_figures.py mag-rrup-scatter $imdb_ffp $out_dir

### Model Development Dataset Magnitude and Distance Histograms
echo "Generating model development dataset magnitude and distance histograms..."
export fig_size="6.5,2.25"
python gen_paper_figures.py model-dev-dataset-hist $full_base_model_dir $out_dir --n-bins 15
export fig_size=$default_fig_size

### DS Dataset Magnitude and Distance Histograms
export fig_size="6.5,2.25"
echo "Generating DS dataset magnitude and distance histograms..."
python gen_paper_figures.py ds-dataset-hist $full_base_model_dir $out_dir --n-bins 15
export fig_size=$default_fig_size

# # ### Bias & Residual Standard Deviation
# echo "Generating mixed effects regression results..."
# python gen_paper_figures.py mera-model-bias-std $cv_base_model_dir $cv_location_model_dir $out_dir --full-base-model-results-dir $full_base_model_dir --full-loc-model-results-dir $full_location_model_dir

# ### Event-level Bias & Residual Standard Deviation
# echo "Generating event-level bias and residual standard deviation results..."
# python gen_paper_figures.py mera-model-bias-std $cv_base_model_dir $cv_location_model_dir $out_dir --event-mera --show-sim

# ### Basin Site Term Comparison
echo "Generating basin site term comparison..."
python gen_paper_figures.py mera-basin-site-term $cv_base_model_dir $cv_location_model_dir $out_dir --min-period 0.5

# # ### Site-to-site residual distribution
# # echo "Generating site-to-site residual distribution...."
# # python gen_paper_figures.py site-to-site-residual-hist pSA_0.01 $cv_base_model_dir $cv_location_model_dir $out_dir --y-max-limit 8 --empty-xaxis
# # python gen_paper_figures.py site-to-site-residual-hist pSA_0.1 $cv_base_model_dir $cv_location_model_dir $out_dir --y-max-limit 8 --empty-xaxis --empty-yaxis
# export fig_size="3.25,3"
# python gen_paper_figures.py site-to-site-residual-hist pSA_3.0 $cv_base_model_dir $cv_location_model_dir $out_dir --y-max-limit 7 
# python gen_paper_figures.py site-to-site-residual-hist pSA_5.0 $cv_base_model_dir $cv_location_model_dir $out_dir --y-max-limit 7 --empty-yaxis    
# export fig_size=$default_fig_size

# ### Site Term Maps
# echo "Generating site term maps..."
# python gen_paper_figures.py nn-gmm-site-term-map $cv_base_model_dir $out_dir pSA_5.0 base --grid-spacing 50e/50e
# python gen_paper_figures.py nn-gmm-site-term-map $cv_location_model_dir $out_dir pSA_5.0 loc --grid-spacing 50e/50e

# ### Model Trends
# echo "Generating model trend figures..."
# export fig_size="3.25,3"
# python gen_paper_figures.py model-trends $full_base_model_dir $full_location_model_dir $SCRIPT_DIR/../configs/figure_configs/mag_crustal_shortDist_modelTrend.yaml $out_dir pSA_0.5 pSA_1.0 pSA_5.0 pSA_10.0 --locations HORC --locations LHUS --no-cv
# python gen_paper_figures.py model-trends $full_base_model_dir $full_location_model_dir $SCRIPT_DIR/../configs/figure_configs/rrup_crustal_mag6p5_modelTrend.yaml $out_dir pSA_0.5 pSA_1.0 pSA_5.0 pSA_10.0 --locations HORC --locations LHUS --no-cv
# export fig_size=$default_fig_size

### Seismic Hazard Curves (tidy up!!)
## HORC
python gen_paper_figures.py single-site-hazard-plot HORC pSA_3.0 $full_base_model_dir/ds_hazard/ref_sites $full_location_model_dir/ds_hazard/ref_sites $emp_ds_hazard_results/ref_sites $out_dir
python gen_paper_figures.py single-site-ds-hazard-plot HORC pSA_3.0 $full_base_model_dir/ds_hazard/ref_sites $full_location_model_dir/ds_hazard/ref_sites $emp_ds_hazard_results/ref_sites $out_dir

## 4-Plot
# export fig_size="3.25,3"
# python gen_paper_figures.py site-hazard-4-plot $full_base_model_dir/ds_hazard/ref_sites $full_location_model_dir/ds_hazard/ref_sites $emp_ds_hazard_results/ref_sites $out_dir
# export fig_size=$default_fig_size

# 2-Plot
export fig_size="3.25,3"
python gen_paper_figures.py site-hazard-2-plot $full_base_model_dir/ds_hazard/ref_sites $full_location_model_dir/ds_hazard/ref_sites $emp_ds_hazard_results/ref_sites $out_dir
export fig_size=$default_fig_size

### UHS Curves
export fig_size="3.25,2.5"
python gen_paper_figures.py site-uhs-plot $full_base_model_dir/ds_hazard/ref_sites $full_location_model_dir/ds_hazard/ref_sites $emp_ds_hazard_results/ref_sites $out_dir
export fig_size=$default_fig_size

# ### DS Hazard ratio maps
# echo "Generating DS hazard ratio maps for ln(emp/base) at RP=2475"
# python gen_paper_figures.py hazard-ratio-map $emp_ds_hazard_results/uniform_grid $full_base_model_dir/ds_hazard/uniform_grid $imdb_ffp $out_dir pSA_5.0 2475 "ln(Empirical/Base Model)" ratio_emp_base 250e/250e  --cb-max 1.0
# python gen_paper_figures.py hazard-ratio-map $emp_ds_hazard_results/uniform_grid $full_base_model_dir/ds_hazard/uniform_grid $imdb_ffp $out_dir pSA_0.5 2475 "ln(Empirical/Base Model)" ratio_emp_base 250e/250e  --cb-max 1.0

# echo "Generating DS hazard ratio maps for ln(base/loc) at RP=2475"
# python gen_paper_figures.py hazard-ratio-map $full_base_model_dir/ds_hazard/uniform_grid $full_location_model_dir/ds_hazard/uniform_grid $imdb_ffp $out_dir pSA_5.0 2475 "ln(Base Model/Location Model)" ratio_base_loc 250e/250e --cb-max 1.0
# python gen_paper_figures.py hazard-ratio-map $full_base_model_dir/ds_hazard/uniform_grid $full_location_model_dir/ds_hazard/uniform_grid $imdb_ffp $out_dir pSA_0.5 2475 "ln(Base Model/Location Model)" ratio_base_loc 250e/250e --cb-max 1.0

# # ### Total hazard maps (CS fault + DS)
# echo "Generating total hazard maps for pSA(5.0s) and RP=2475"
# python gen_paper_figures.py total-hazard-map $imdb_ffp $full_base_model_dir/ds_hazard/uniform_grid $out_dir pSA_5.0 2475 "CS 25.6 + Base Model" total_base 250e/250e  
# python gen_paper_figures.py total-hazard-map $imdb_ffp $full_location_model_dir/ds_hazard/uniform_grid $out_dir pSA_5.0 2475 "CS 25.6 + Location Model" total_loc 250e/250e 
# python gen_paper_figures.py total-hazard-map $imdb_ffp $emp_ds_hazard_results/uniform_grid $out_dir pSA_5.0 2475 "CS 25.6 + Empirical Logic Tree" total_emp 250e/250e  

# # # echo "Generating total hazard maps for pSA(0.5s) and RP=2475"
# python gen_paper_figures.py total-hazard-map $imdb_ffp $full_base_model_dir/ds_hazard/uniform_grid $out_dir pSA_0.5 2475 "CS 25.6 + Base Model" total_base 250e/250e  
# python gen_paper_figures.py total-hazard-map $imdb_ffp $full_location_model_dir/ds_hazard/uniform_grid $out_dir pSA_0.5 2475 "CS 25.6 + Location Model" total_loc 250e/250e  
# python gen_paper_figures.py total-hazard-map $imdb_ffp $emp_ds_hazard_results/uniform_grid $out_dir pSA_0.5 2475 "CS 25.6 + Empirical Logic Tree" total_emp 250e/250e  



# --------------------------------------- Electronic Supplementary Material Figures ---------------------------------------

# # Sample weighting figures
# export fig_size="3.25,3"
# python gen_paper_figures.py sample-weights $imdb_ffp $test_events_ffp $out_dir
# export fig_size=$default_fig_size

# # Site distribution maps
# python gen_paper_figures.py site-distribution-maps $imdb_ffp $out_dir

# # # DS Hazard ratio maps
# # echo "Generating DS hazard ratio maps for ln(emp/base) at RP=475"
# # python gen_paper_figures.py hazard-ratio-map $emp_ds_hazard_results/uniform_grid $full_base_model_dir/ds_hazard/uniform_grid $imdb_ffp $out_dir pSA_5.0 475 "ln(Empirical/Base Model)" ratio_emp_base 100e/100e  --cb-max 1.0
# # python gen_paper_figures.py hazard-ratio-map $emp_ds_hazard_results/uniform_grid $full_base_model_dir/ds_hazard/uniform_grid $imdb_ffp $out_dir pSA_0.5 475 "ln(Empirical/Base Model)" ratio_emp_base 100e/100e  --cb-max 1.0

# # echo "Generating DS hazard ratio maps for ln(base/loc) at RP=475"
# # python gen_paper_figures.py hazard-ratio-map $full_base_model_dir/ds_hazard/uniform_grid $full_location_model_dir/ds_hazard/uniform_grid $imdb_ffp $out_dir pSA_5.0 475 "ln(Base Model/Location Model)" ratio_emp_loc 100e/100e  --cb-max 1.0
# # python gen_paper_figures.py hazard-ratio-map $full_base_model_dir/ds_hazard/uniform_grid $full_location_model_dir/ds_hazard/uniform_grid $imdb_ffp $out_dir pSA_0.5 475 "ln(Base Model/Location Model)" ratio_emp_loc 100e/100e  --cb-max 1.0


# ### Model Trends
# echo "Generating ES model trend figures..."
# export fig_size="3.25,3"
# # Crustal - Magnitude - Full
# python gen_paper_figures.py model-trends $full_base_model_dir $full_location_model_dir $SCRIPT_DIR/../configs/figure_configs/mag_crustal_shortDist_modelTrend.yaml $out_dir pSA_0.5 pSA_1.0 pSA_5.0 pSA_10.0 --locations HORC --locations LHUS --no-cv
# python gen_paper_figures.py model-trends $full_base_model_dir $full_location_model_dir $SCRIPT_DIR/../configs/figure_configs/mag_crustal_moderateDist_modelTrend.yaml $out_dir pSA_0.5 pSA_1.0 pSA_5.0 pSA_10.0 --locations HORC --locations LHUS --no-cv
# python gen_paper_figures.py model-trends $full_base_model_dir $full_location_model_dir $SCRIPT_DIR/../configs/figure_configs/mag_crustal_largeDist_modelTrend.yaml $out_dir pSA_0.5 pSA_1.0 pSA_5.0 pSA_10.0 --locations HORC --locations LHUS --no-cv

# # Slab - Magnitude - Full
# python gen_paper_figures.py model-trends $full_base_model_dir $full_location_model_dir $SCRIPT_DIR/../configs/figure_configs/mag_slab_shortDist_modelTrend.yaml $out_dir pSA_0.5 pSA_1.0 pSA_5.0 pSA_10.0 --locations HORC --locations LHUS --no-cv
# python gen_paper_figures.py model-trends $full_base_model_dir $full_location_model_dir $SCRIPT_DIR/../configs/figure_configs/mag_slab_moderateDist_modelTrend.yaml $out_dir pSA_0.5 pSA_1.0 pSA_5.0 pSA_10.0 --locations HORC --locations LHUS --no-cv
# python gen_paper_figures.py model-trends $full_base_model_dir $full_location_model_dir $SCRIPT_DIR/../configs/figure_configs/mag_slab_largeDist_modelTrend.yaml $out_dir pSA_0.5 pSA_1.0 pSA_5.0 pSA_10.0 --locations HORC --locations LHUS --no-cv


# # Rrup - Full
# python gen_paper_figures.py model-trends $full_base_model_dir $full_location_model_dir $SCRIPT_DIR/../configs/figure_configs/rrup_crustal_mag6p5_modelTrend.yaml $out_dir pSA_0.5 pSA_1.0 pSA_5.0 pSA_10.0 --locations HORC --locations LHUS --no-cv
# python gen_paper_figures.py model-trends $full_base_model_dir $full_location_model_dir $SCRIPT_DIR/../configs/figure_configs/rrup_crustal_mag7p25_modelTrend.yaml $out_dir pSA_0.5 pSA_1.0 pSA_5.0 pSA_10.0 --locations HORC --locations LHUS --no-cv
# python gen_paper_figures.py model-trends $full_base_model_dir $full_location_model_dir $SCRIPT_DIR/../configs/figure_configs/rrup_interface_mag8p0_modelTrend.yaml $out_dir pSA_0.5 pSA_1.0 pSA_5.0 pSA_10.0 --locations HORC --locations LHUS --no-cv
# export fig_size=$default_fig_size



exit

# ------------------------------ Archive -------------------------

### Magnitude-Tectonic Type Distribution 
# python gen_paper_figures.py mag-tect-type-dist $imdb_ffp $out_dir

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

### Seismic Hazard Curves (tidy up!!)
## LHUS
# python gen_paper_figures.py single-site-hazard-plot LHUS pSA_3.0 $full_base_model_dir/ds_hazard/ref_sites $full_location_model_dir/ds_hazard/ref_sites $emp_ds_hazard_results/ref_sites $out_dir
# python gen_paper_figures.py single-site-ds-hazard-plot LHUS pSA_3.0 $full_base_model_dir/ds_hazard/ref_sites $full_location_model_dir/ds_hazard/ref_sites $emp_ds_hazard_results/ref_sites $out_dir

## MGCS
# python gen_paper_figures.py single-site-hazard-plot MGCS pSA_3.0 $full_base_model_dir/ds_hazard/ref_sites $full_location_model_dir/ds_hazard/ref_sites $emp_ds_hazard_results/ref_sites $out_dir
# python gen_paper_figures.py single-site-ds-hazard-plot MGCS pSA_3.0 $full_base_model_dir/ds_hazard/ref_sites $full_location_model_dir/ds_hazard/ref_sites $emp_ds_hazard_results/ref_sites $out_dir