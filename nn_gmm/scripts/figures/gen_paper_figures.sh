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

empdb_ffp=$wdata/nn_gmm/20251222_CS200_DSSlab_rotd50_empdb.duckdb
echo "empdb_ffp: $empdb_ffp"

test_events_ffp=$wdata/nn_gmm/test_events.npy
echo "test_events_ffp: $test_events_ffp"

emp_ds_hazard_results=$wdata/nn_gmm/results/emp_ds_hazard
echo "emp_ds_hazard_results: $emp_ds_hazard_results"

emp_flt_hazard_results=$wdata/nn_gmm/results/emp_flt_hazard/fault_emp_hazard_site_grid_0.pkl
echo "emp_flt_hazard_results: $emp_flt_hazard_results"

cv_base_model_dir=$wdata/nn_gmm/results/0922_1644_cv_baseV6_25Epochs_6E6Folds
echo "cv_base_model_dir: $cv_base_model_dir"

full_base_model_dir=$wdata/nn_gmm/results/0922_1959_full_baseV6_10Epochs
echo "full_base_model_dir: $full_base_model_dir"

cv_location_model_dir=$wdata/nn_gmm/results/0923_0730_cv_baseV6_locAdjV4_25Epochs_6E6Folds
echo "cv_location_model_dir: $cv_location_model_dir"

full_location_model_dir=$wdata/nn_gmm/results/0923_1156_full_baseV6_locAdjV4_10Epochs
echo "full_location_model_dir: $full_location_model_dir"

out_dir=/Users/claudy/dev/tmp_share/nn_gmm/20260926_paper_figures/v6
# out_dir=/Users/claudy/dev/work/tmp/nn_gmm/test
echo "out_dir: $out_dir"

## Fault Map
python gen_paper_figures.py fault-map $out_dir $imdb_ffp

## Magnitude-Rrrup Scatter Plot
python gen_paper_figures.py mag-rrup-scatter $imdb_ffp $out_dir

## Model Development Dataset Magnitude and Distance Histograms
echo "Generating model development dataset magnitude and distance histograms..."
export fig_size="6.5,2.25"
python gen_paper_figures.py model-dev-dataset-hist $full_base_model_dir $out_dir --n-bins 15
export fig_size=$default_fig_size

### DS Dataset Magnitude and Distance Histograms
export fig_size="6.5,2.25"
echo "Generating DS dataset magnitude and distance histograms..."
python gen_paper_figures.py ds-dataset-hist $full_base_model_dir $out_dir --n-bins 15
export fig_size=$default_fig_size

### Bias & Residual Standard Deviation
echo "Generating mixed effects regression results..."
python gen_paper_figures.py mera-model-bias-std $cv_base_model_dir $cv_location_model_dir $out_dir --full-base-model-results-dir $full_base_model_dir --full-loc-model-results-dir $full_location_model_dir

### Event-level Bias & Residual Standard Deviation
echo "Generating event-level bias and residual standard deviation results..."
python gen_paper_figures.py mera-model-bias-std $cv_base_model_dir $cv_location_model_dir $out_dir --event-mera --show-sim

# Basin Site Term Comparison
echo "Generating basin site term comparison..."
python gen_paper_figures.py mera-basin-site-term $cv_base_model_dir $cv_location_model_dir $out_dir --min-period 0.5

## Site-to-site residual distribution
echo "Generating site-to-site residual distribution...."
export fig_size="3.25,3"
python gen_paper_figures.py site-to-site-residual-hist pSA_3.0 $cv_base_model_dir $cv_location_model_dir $out_dir --y-max-limit 8
python gen_paper_figures.py site-to-site-residual-hist pSA_5.0 $cv_base_model_dir $cv_location_model_dir $out_dir --y-max-limit 8 --empty-yaxis    
export fig_size=$default_fig_size

## Site Term Maps
echo "Generating site term maps..."
python gen_paper_figures.py nn-gmm-site-term-map $cv_base_model_dir $out_dir pSA_5.0 base --grid-spacing 250e/250e
python gen_paper_figures.py nn-gmm-site-term-map $cv_location_model_dir $out_dir pSA_5.0 loc --grid-spacing 250e/250e

### Model Trends
echo "Generating model trend figures..."
export fig_size="3.25,3"
python gen_paper_figures.py model-trends $full_base_model_dir $full_location_model_dir $SCRIPT_DIR/../configs/figure_configs/mag_crustal_shortDist_modelTrend.yaml $out_dir pSA_0.5 pSA_1.0 pSA_5.0 pSA_10.0 --locations HORC --locations LHUS --no-cv
python gen_paper_figures.py model-trends $full_base_model_dir $full_location_model_dir $SCRIPT_DIR/../configs/figure_configs/rrup_crustal_mag6p5_modelTrend.yaml $out_dir pSA_0.5 pSA_1.0 pSA_5.0 pSA_10.0 --locations HORC --locations LHUS --no-cv
export fig_size=$default_fig_size

### Seismic Hazard Curves 
## HORC
python gen_paper_figures.py single-site-hazard-plot HORC pSA_3.0 $full_base_model_dir/ds_hazard/ref_sites $full_location_model_dir/ds_hazard/ref_sites $emp_ds_hazard_results/ref_sites $out_dir --lower-base-ds-dir $full_base_model_dir/ds_hazard/ref_sites_q0p05 --upper-base-ds-dir $full_base_model_dir/ds_hazard/ref_sites_q0p95 
python gen_paper_figures.py single-site-ds-hazard-plot HORC pSA_3.0 $full_base_model_dir/ds_hazard/ref_sites $full_location_model_dir/ds_hazard/ref_sites $emp_ds_hazard_results/ref_sites $out_dir --lower-base-ds-dir $full_base_model_dir/ds_hazard/ref_sites_q0p05 --upper-base-ds-dir $full_base_model_dir/ds_hazard/ref_sites_q0p95

## 2-Plot
export fig_size="3.25,3"
python gen_paper_figures.py site-hazard-2-plot $full_base_model_dir/ds_hazard/ref_sites $full_location_model_dir/ds_hazard/ref_sites $emp_ds_hazard_results/ref_sites $out_dir
python gen_paper_figures.py site-hazard-2-plot $full_base_model_dir/ds_hazard/ref_sites $full_location_model_dir/ds_hazard/ref_sites $emp_ds_hazard_results/ref_sites $out_dir --im pSA_0.01 --y-min 0.00001 --y-max 2 --no-poe-label-right --legend-loc "lower left" --site-label-x 0.5
export fig_size=$default_fig_size

### DS Hazard ratio maps
echo "Generating DS hazard ratio maps for ln(emp/base) at RP=2475"
python gen_paper_figures.py hazard-ratio-map $emp_ds_hazard_results/uniform_grid $full_base_model_dir/ds_hazard/uniform_grid $imdb_ffp $out_dir pSA_5.0 2475 "ln(Empirical/Base Model)" ratio_emp_base 250e/250e  --cb-max 1.5
python gen_paper_figures.py hazard-ratio-map $emp_ds_hazard_results/uniform_grid $full_base_model_dir/ds_hazard/uniform_grid $imdb_ffp $out_dir pSA_0.5 2475 "ln(Empirical/Base Model)" ratio_emp_base 250e/250e  --cb-max 1.5

echo "Generating DS hazard ratio maps for ln(base/loc) at RP=2475"
python gen_paper_figures.py hazard-ratio-map $full_base_model_dir/ds_hazard/uniform_grid $full_location_model_dir/ds_hazard/uniform_grid $imdb_ffp $out_dir pSA_5.0 2475 "ln(Base Model/Location Model)" ratio_base_loc 250e/250e --cb-max 1.5
python gen_paper_figures.py hazard-ratio-map $full_base_model_dir/ds_hazard/uniform_grid $full_location_model_dir/ds_hazard/uniform_grid $imdb_ffp $out_dir pSA_0.5 2475 "ln(Base Model/Location Model)" ratio_base_loc 250e/250e --cb-max 1.5

# ### Total hazard maps (CS fault + DS)


# # --------------------------------------- Electronic Supplement Figures ---------------------------------------

export fig_dpi="300"

# Sample weighting figures
export fig_size="3.25,3"
python gen_paper_figures.py sample-weights $imdb_ffp $test_events_ffp $out_dir
export fig_size=$default_fig_size

# Site distribution maps
python gen_paper_figures.py site-distribution-maps $imdb_ffp $out_dir

### UHS Curves
export fig_size="3.25,2.5"
python gen_paper_figures.py site-uhs-plot $full_base_model_dir/ds_hazard/ref_sites $full_location_model_dir/ds_hazard/ref_sites $emp_ds_hazard_results/ref_sites $out_dir
export fig_size=$default_fig_size

# ### DS Hazard ratio maps
echo "Generating DS hazard ratio maps for ln(emp/base) at RP=2475"
python gen_paper_figures.py hazard-ratio-map $emp_ds_hazard_results/uniform_grid $full_base_model_dir/ds_hazard/uniform_grid $imdb_ffp $out_dir pSA_5.0 475 "ln(Empirical/Base Model)" ratio_emp_base 250e/250e  --cb-max 1.5
python gen_paper_figures.py hazard-ratio-map $emp_ds_hazard_results/uniform_grid $full_base_model_dir/ds_hazard/uniform_grid $imdb_ffp $out_dir pSA_0.5 475 "ln(Empirical/Base Model)" ratio_emp_base 250e/250e  --cb-max 1.5
python gen_paper_figures.py hazard-ratio-map $emp_ds_hazard_results/uniform_grid $full_base_model_dir/ds_hazard/uniform_grid $imdb_ffp $out_dir pSA_0.01 475 "ln(Empirical/Base Model)" ratio_emp_base 250e/250e  --cb-max 1.5
python gen_paper_figures.py hazard-ratio-map $emp_ds_hazard_results/uniform_grid $full_base_model_dir/ds_hazard/uniform_grid $imdb_ffp $out_dir pSA_0.01 2475 "ln(Empirical/Base Model)" ratio_emp_base 250e/250e  --cb-max 1.5

echo "Generating DS hazard ratio maps for ln(base/loc) at RP=475"
python gen_paper_figures.py hazard-ratio-map $full_base_model_dir/ds_hazard/uniform_grid $full_location_model_dir/ds_hazard/uniform_grid $imdb_ffp $out_dir pSA_5.0 475 "ln(Base Model/Location Model)" ratio_base_loc 250e/250e  --cb-max 1.5
python gen_paper_figures.py hazard-ratio-map $full_base_model_dir/ds_hazard/uniform_grid $full_location_model_dir/ds_hazard/uniform_grid $imdb_ffp $out_dir pSA_0.5 475 "ln(Base Model/Location Model)" ratio_base_loc 250e/250e  --cb-max 1.5
python gen_paper_figures.py hazard-ratio-map $full_base_model_dir/ds_hazard/uniform_grid $full_location_model_dir/ds_hazard/uniform_grid $imdb_ffp $out_dir pSA_0.01 475 "ln(Base Model/Location Model)" ratio_base_loc 250e/250e  --cb-max 1.5
python gen_paper_figures.py hazard-ratio-map $full_base_model_dir/ds_hazard/uniform_grid $full_location_model_dir/ds_hazard/uniform_grid $imdb_ffp $out_dir pSA_0.01 2475 "ln(Base Model/Location Model)" ratio_base_loc 250e/250e  --cb-max 1.5

### Total hazard maps (CS fault + DS)
echo "Generating total hazard maps for pSA(5.0s) and RP=475"
python gen_paper_figures.py total-hazard-map $imdb_ffp $full_base_model_dir/ds_hazard/uniform_grid $out_dir pSA_5.0 475 "CS 25.6 + Base Model" total_base 250e/250e  
python gen_paper_figures.py total-hazard-map $imdb_ffp $full_location_model_dir/ds_hazard/uniform_grid $out_dir pSA_5.0 475 "CS 25.6 + Location Model" total_loc 250e/250e 
python gen_paper_figures.py total-hazard-map $imdb_ffp $emp_ds_hazard_results/uniform_grid $out_dir pSA_5.0 475 "CS 25.6 + Empirical Logic Tree" total_emp 250e/250e  

echo "Generating total hazard maps for pSA(0.5s) and RP=475"
python gen_paper_figures.py total-hazard-map $imdb_ffp $full_base_model_dir/ds_hazard/uniform_grid $out_dir pSA_0.5 475 "CS 25.6 + Base Model" total_base 250e/250e  
python gen_paper_figures.py total-hazard-map $imdb_ffp $full_location_model_dir/ds_hazard/uniform_grid $out_dir pSA_0.5 475 "CS 25.6 + Location Model" total_loc 250e/250e  
python gen_paper_figures.py total-hazard-map $imdb_ffp $emp_ds_hazard_results/uniform_grid $out_dir pSA_0.5 475 "CS 25.6 + Empirical Logic Tree" total_emp 250e/250e  

echo "Generating total hazard maps for pSA(0.01s) and RP=475"
python gen_paper_figures.py total-hazard-map $imdb_ffp $full_base_model_dir/ds_hazard/uniform_grid $out_dir pSA_0.01 475 "CS 25.6 + Base Model" total_base 250e/250e  
python gen_paper_figures.py total-hazard-map $imdb_ffp $full_location_model_dir/ds_hazard/uniform_grid $out_dir pSA_0.01 475 "CS 25.6 + Location Model" total_loc 250e/250e  
python gen_paper_figures.py total-hazard-map $imdb_ffp $emp_ds_hazard_results/uniform_grid $out_dir pSA_0.01 475 "CS 25.6 + Empirical Logic Tree" total_emp 250e/250e  

echo "Generating total hazard maps for pSA(5.0s) and RP=2475"
python gen_paper_figures.py total-hazard-map $imdb_ffp $full_base_model_dir/ds_hazard/uniform_grid $out_dir pSA_5.0 2475 "CS 25.6 + Base Model" total_base 250e/250e  
python gen_paper_figures.py total-hazard-map $imdb_ffp $full_location_model_dir/ds_hazard/uniform_grid $out_dir pSA_5.0 2475 "CS 25.6 + Location Model" total_loc 250e/250e 
python gen_paper_figures.py total-hazard-map $imdb_ffp $emp_ds_hazard_results/uniform_grid $out_dir pSA_5.0 2475 "CS 25.6 + Empirical Logic Tree" total_emp 250e/250e  

echo "Generating total hazard maps for pSA(0.5s) and RP=2475"
python gen_paper_figures.py total-hazard-map $imdb_ffp $full_base_model_dir/ds_hazard/uniform_grid $out_dir pSA_0.5 2475 "CS 25.6 + Base Model" total_base 250e/250e  
python gen_paper_figures.py total-hazard-map $imdb_ffp $full_location_model_dir/ds_hazard/uniform_grid $out_dir pSA_0.5 2475 "CS 25.6 + Location Model" total_loc 250e/250e  
python gen_paper_figures.py total-hazard-map $imdb_ffp $emp_ds_hazard_results/uniform_grid $out_dir pSA_0.5 2475 "CS 25.6 + Empirical Logic Tree" total_emp 250e/250e  

echo "Generating total hazard maps for pSA(0.01s) and RP=2475"
python gen_paper_figures.py total-hazard-map $imdb_ffp $full_base_model_dir/ds_hazard/uniform_grid $out_dir pSA_0.01 2475 "CS 25.6 + Base Model" total_base 250e/250e  
python gen_paper_figures.py total-hazard-map $imdb_ffp $full_location_model_dir/ds_hazard/uniform_grid $out_dir pSA_0.01 2475 "CS 25.6 + Location Model" total_loc 250e/250e  
python gen_paper_figures.py total-hazard-map $imdb_ffp $emp_ds_hazard_results/uniform_grid $out_dir pSA_0.01 2475 "CS 25.6 + Empirical Logic Tree" total_emp 250e/250e  



# ### Site Term Maps
echo "Generating site term maps..."
python gen_paper_figures.py nn-gmm-site-term-map $cv_base_model_dir $out_dir pSA_0.01 base --grid-spacing 250e/250e
python gen_paper_figures.py nn-gmm-site-term-map $cv_location_model_dir $out_dir pSA_0.01 loc --grid-spacing 250e/250e

python gen_paper_figures.py nn-gmm-site-term-map $cv_base_model_dir $out_dir pSA_0.1 base --grid-spacing 250e/250e
python gen_paper_figures.py nn-gmm-site-term-map $cv_location_model_dir $out_dir pSA_0.1 loc --grid-spacing 250e/250e

python gen_paper_figures.py nn-gmm-site-term-map $cv_base_model_dir $out_dir pSA_0.5 base --grid-spacing 250e/250e
python gen_paper_figures.py nn-gmm-site-term-map $cv_location_model_dir $out_dir pSA_0.5 loc --grid-spacing 250e/250e

python gen_paper_figures.py nn-gmm-site-term-map $cv_base_model_dir $out_dir pSA_1.0 base --grid-spacing 250e/250e
python gen_paper_figures.py nn-gmm-site-term-map $cv_location_model_dir $out_dir pSA_1.0 loc --grid-spacing 250e/250e

python gen_paper_figures.py nn-gmm-site-term-map $cv_base_model_dir $out_dir pSA_2.5 base --grid-spacing 250e/250e
python gen_paper_figures.py nn-gmm-site-term-map $cv_location_model_dir $out_dir pSA_2.5 loc --grid-spacing 250e/250e

python gen_paper_figures.py nn-gmm-site-term-map $cv_base_model_dir $out_dir pSA_10.0 base --grid-spacing 250e/250e
python gen_paper_figures.py nn-gmm-site-term-map $cv_location_model_dir $out_dir pSA_10.0 loc --grid-spacing 250e/250e

### Model Trends
echo "Generating ES model trend figures..."
export fig_size="3.25,3"
# Crustal - Magnitude - Full
python gen_paper_figures.py model-trends $full_base_model_dir $full_location_model_dir $SCRIPT_DIR/../configs/figure_configs/mag_crustal_shortDist_modelTrend.yaml $out_dir pSA_0.5 pSA_1.0 pSA_5.0 pSA_10.0 --locations HORC --locations LHUS --no-cv
python gen_paper_figures.py model-trends $full_base_model_dir $full_location_model_dir $SCRIPT_DIR/../configs/figure_configs/mag_crustal_moderateDist_modelTrend.yaml $out_dir pSA_0.5 pSA_1.0 pSA_5.0 pSA_10.0 --locations HORC --locations LHUS --no-cv
python gen_paper_figures.py model-trends $full_base_model_dir $full_location_model_dir $SCRIPT_DIR/../configs/figure_configs/mag_crustal_largeDist_modelTrend.yaml $out_dir pSA_0.5 pSA_1.0 pSA_5.0 pSA_10.0 --locations HORC --locations LHUS --no-cv

# Slab - Magnitude - Full
python gen_paper_figures.py model-trends $full_base_model_dir $full_location_model_dir $SCRIPT_DIR/../configs/figure_configs/mag_slab_shortDist_modelTrend.yaml $out_dir pSA_0.5 pSA_1.0 pSA_5.0 pSA_10.0 --locations HORC --locations LHUS --no-cv
python gen_paper_figures.py model-trends $full_base_model_dir $full_location_model_dir $SCRIPT_DIR/../configs/figure_configs/mag_slab_moderateDist_modelTrend.yaml $out_dir pSA_0.5 pSA_1.0 pSA_5.0 pSA_10.0 --locations HORC --locations LHUS --no-cv
python gen_paper_figures.py model-trends $full_base_model_dir $full_location_model_dir $SCRIPT_DIR/../configs/figure_configs/mag_slab_largeDist_modelTrend.yaml $out_dir pSA_0.5 pSA_1.0 pSA_5.0 pSA_10.0 --locations HORC --locations LHUS --no-cv

# Rrup - Full
python gen_paper_figures.py model-trends $full_base_model_dir $full_location_model_dir $SCRIPT_DIR/../configs/figure_configs/rrup_crustal_mag6p5_modelTrend.yaml $out_dir pSA_0.5 pSA_1.0 pSA_5.0 pSA_10.0 --locations HORC --locations LHUS --no-cv
python gen_paper_figures.py model-trends $full_base_model_dir $full_location_model_dir $SCRIPT_DIR/../configs/figure_configs/rrup_crustal_mag7p25_modelTrend.yaml $out_dir pSA_0.5 pSA_1.0 pSA_5.0 pSA_10.0 --locations HORC --locations LHUS --no-cv
python gen_paper_figures.py model-trends $full_base_model_dir $full_location_model_dir $SCRIPT_DIR/../configs/figure_configs/rrup_interface_mag8p0_modelTrend.yaml $out_dir pSA_0.5 pSA_1.0 pSA_5.0 pSA_10.0 --locations HORC --locations LHUS --no-cv
export fig_size=$default_fig_size

### Standard devaiation of normalized residuals
python gen_paper_figures.py standardized-residuals $cv_base_model_dir $cv_location_model_dir $empdb_ffp  $out_dir

## Disaggregation plots
python plot_cmds.py plot-disagg $full_base_model_dir/disagg/HORC_pSA_0p01_RP2475_CS_EMP_disagg.parquet $out_dir HORC pSA_0.01 2475
python plot_cmds.py plot-disagg $full_base_model_dir/disagg/HORC_pSA_0p01_RP2475_CS_NN_disagg.parquet $out_dir HORC pSA_0.01 2475
python plot_cmds.py plot-disagg $full_base_model_dir/disagg/HORC_pSA_0p01_RP475_CS_EMP_disagg.parquet $out_dir HORC pSA_0.01 475
python plot_cmds.py plot-disagg $full_base_model_dir/disagg/HORC_pSA_0p01_RP475_CS_NN_disagg.parquet $out_dir HORC pSA_0.01 475
python plot_cmds.py plot-disagg $full_base_model_dir/disagg/HORC_pSA_5p0_RP2475_CS_EMP_disagg.parquet $out_dir HORC pSA_5.0 2475
python plot_cmds.py plot-disagg $full_base_model_dir/disagg/HORC_pSA_5p0_RP2475_CS_NN_disagg.parquet $out_dir HORC pSA_5.0 2475
python plot_cmds.py plot-disagg $full_base_model_dir/disagg/HORC_pSA_5p0_RP475_CS_EMP_disagg.parquet $out_dir HORC pSA_5.0 475
python plot_cmds.py plot-disagg $full_base_model_dir/disagg/HORC_pSA_5p0_RP475_CS_NN_disagg.parquet $out_dir HORC pSA_5.0 475
python plot_cmds.py plot-disagg $full_base_model_dir/disagg/LHUS_pSA_0p01_RP2475_CS_EMP_disagg.parquet $out_dir LHUS pSA_0.01 2475
python plot_cmds.py plot-disagg $full_base_model_dir/disagg/LHUS_pSA_0p01_RP2475_CS_NN_disagg.parquet $out_dir LHUS pSA_0.01 2475
python plot_cmds.py plot-disagg $full_base_model_dir/disagg/LHUS_pSA_0p01_RP475_CS_EMP_disagg.parquet $out_dir LHUS pSA_0.01 475
python plot_cmds.py plot-disagg $full_base_model_dir/disagg/LHUS_pSA_0p01_RP475_CS_NN_disagg.parquet $out_dir LHUS pSA_0.01 475
python plot_cmds.py plot-disagg $full_base_model_dir/disagg/LHUS_pSA_5p0_RP2475_CS_EMP_disagg.parquet $out_dir LHUS pSA_5.0 2475
python plot_cmds.py plot-disagg $full_base_model_dir/disagg/LHUS_pSA_5p0_RP2475_CS_NN_disagg.parquet $out_dir LHUS pSA_5.0 2475
python plot_cmds.py plot-disagg $full_base_model_dir/disagg/LHUS_pSA_5p0_RP475_CS_EMP_disagg.parquet $out_dir LHUS pSA_5.0 475
python plot_cmds.py plot-disagg $full_base_model_dir/disagg/LHUS_pSA_5p0_RP475_CS_NN_disagg.parquet $out_dir LHUS pSA_5.0 475

# Combine empirical (left) and surrogate (right) disagg plots
python gen_paper_figures.py combine-disagg-figures $out_dir/HORC_pSA_0p01_RP475_CS_EMP_disagg_disagg_plot.png $out_dir/HORC_pSA_0p01_RP475_CS_NN_disagg_disagg_plot.png $out_dir/HORC_pSA_0p01_RP475_disagg_combined.png
python gen_paper_figures.py combine-disagg-figures $out_dir/HORC_pSA_0p01_RP2475_CS_EMP_disagg_disagg_plot.png $out_dir/HORC_pSA_0p01_RP2475_CS_NN_disagg_disagg_plot.png $out_dir/HORC_pSA_0p01_RP2475_disagg_combined.png
python gen_paper_figures.py combine-disagg-figures $out_dir/HORC_pSA_5p0_RP475_CS_EMP_disagg_disagg_plot.png $out_dir/HORC_pSA_5p0_RP475_CS_NN_disagg_disagg_plot.png $out_dir/HORC_pSA_5p0_RP475_disagg_combined.png
python gen_paper_figures.py combine-disagg-figures $out_dir/HORC_pSA_5p0_RP2475_CS_EMP_disagg_disagg_plot.png $out_dir/HORC_pSA_5p0_RP2475_CS_NN_disagg_disagg_plot.png $out_dir/HORC_pSA_5p0_RP2475_disagg_combined.png
python gen_paper_figures.py combine-disagg-figures $out_dir/LHUS_pSA_0p01_RP475_CS_EMP_disagg_disagg_plot.png $out_dir/LHUS_pSA_0p01_RP475_CS_NN_disagg_disagg_plot.png $out_dir/LHUS_pSA_0p01_RP475_disagg_combined.png
python gen_paper_figures.py combine-disagg-figures $out_dir/LHUS_pSA_0p01_RP2475_CS_EMP_disagg_disagg_plot.png $out_dir/LHUS_pSA_0p01_RP2475_CS_NN_disagg_disagg_plot.png $out_dir/LHUS_pSA_0p01_RP2475_disagg_combined.png
python gen_paper_figures.py combine-disagg-figures $out_dir/LHUS_pSA_5p0_RP475_CS_EMP_disagg_disagg_plot.png $out_dir/LHUS_pSA_5p0_RP475_CS_NN_disagg_disagg_plot.png $out_dir/LHUS_pSA_5p0_RP475_disagg_combined.png
python gen_paper_figures.py combine-disagg-figures $out_dir/LHUS_pSA_5p0_RP2475_CS_EMP_disagg_disagg_plot.png $out_dir/LHUS_pSA_5p0_RP2475_CS_NN_disagg_disagg_plot.png $out_dir/LHUS_pSA_5p0_RP2475_disagg_combined.png

## Hazard error plots
python gen_paper_figures.py nn-fault-hazard-bias-res-std $cv_base_model_dir/fault_hazard_site_grid_0.pkl $cv_location_model_dir/fault_hazard_site_grid_0.pkl $out_dir --rps 475 --rps 2475 --emp-flt-hazard-results-ffp $emp_flt_hazard_results

### Loss Curves
echo "Generating loss curves..."
python gen_paper_figures.py plot-loss-curves $cv_base_model_dir $out_dir
python gen_paper_figures.py plot-loss-curves $cv_location_model_dir $out_dir

exit

