# Combining Direct Simulation and Surrogate Modelling for Simulation-Consistent Probabilistic Seismic Hazard Analysis

This repository contains the code for reproducing the results from the paper.

Note: It is advised to use a Linux machine. All instructions below assume a Linux OS.
If you encounter any problems, please raise an issue on the GitHub repository.

## Installation

### Prerequisites

The following libraries must be available:
- [GMT](https://docs.generic-mapping-tools.org/dev/index.html) version 6.6

### Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```

2. Install the package:
   ```bash
   pip install ./post_event_gm_gnn
   ```

## Data Setup

### Environment Configuration

Most data loading/saving is done relative to the `wdata` environment variable.

1. Create a data directory for this project (e.g., `base_data_dir`)
2. Add the following to your `.bashrc` (or equivalent):
   ```bash
   export wdata=/path/to/base_data_dir
   ```

### Download Data

Download the data file from [here]() and unzip it.

This will create the following directory structure:

```
base_data_dir
    |--- 20251222_CS200_DSSlab_rotd50_empdb.duckdb
    |--- 20251222_CS200_DSSlab_rotd50_imdb.duckdb
    |--- test_events.npy
    ├── cs200_hazard
    ├── other
    │   ├── dupuis_moderate_mag
    │   ├── dupuis_small_mag
    │   └── lee_nz_validation_2022
    ├── results
       ├── 0613_1752_cv_baseV5p1_25Epochs_6E6Folds
       │   ├── event_mera_site_term
       │   └── mera_site_term
       ├── 0613_2113_full_baseV5p1_10Epochs
       │   ├── ds_hazard
       │   │   ├── ref_sites
       │   │   └── uniform_grid
       │   └── test_mera
       ├── 0613_2142_cv_baseV5p1_locAdjV4_25Epochs_6E6Folds
       │   ├── event_mera_site_term
       │   └── mera_site_term
       ├── 0614_0214_full_baseV5p1_locAdjV4_10Epochs
       │   ├── ds_hazard
       │   │   ├── ref_sites
       │   │   └── uniform_grid
       │   └── test_mera
       ├── emp_ds_hazard
       │   ├── ref_sites
       │   └── uniform_grid
       └── loc_models
           └── 0929_1130_optimized_siteCond_25Epochs
```

## Data Overview

### Databases

- **20251222_CS200_DSSlab_rotd50_imdb.duckdb**: Primary data resource. Contains all rupture, source, and site information, in addition to the simulation IM values. 
- **20251222_CS200_DSSlab_rotd50_empdb.duckdb**: Contains IM distribution parameters for the empirical GM logic tree

### Results

The results folder contains the following

- `0613_1752_cv_baseV5p1_25Epochs_6E6Folds` - Cross-validation (CV) results for the base surrogate model
- `0613_2142_cv_baseV5p1_locAdjV4_25Epochs_6E6Folds` - Cross-validation results for the location-specific surrogate model  
    - Each of these contains:
        - One directory per cross-validation run
        - A `mera_site_term` folder, which contains the mixed-effects regression results 
        - A `event_mera_site_term` folder, which contains the mixed-effects regression results at the event level
        - A `val_results.parquet` file, which is a dataframe of the combined validation results
- `0613_2113_full_baseV5p1_10Epochs` - Full base surrogate model
- `0614_0214_full_baseV5p1_locAdjV4_10Epochs` - Full location-specific surrogate model
    - Each of these contains:
        - `test_mera` folder, which contains the test dataset mixed-effects regression results
        - `test_results.parquet` file, which is a dataframe with test dataset results
        - `ds_hazard` folder, which contains the distributed seismicity hazard results
            - `ref_sites` distributed seismicity hazard at a small set of reference sites (aligned with Cybershake NZ 200m paper)
            - `unfiorm_grid` distributed seismicity hazard on the uniform grid
- `emp_ds_hazard` - Empirical GM logic tree distributed seismicity hazard results
- `loc_models/0929_1130_optimized_siteCond_25Epochs` - Pre-trained location-embedding model


### Other

The `other` folder contains the mixed-effects regression results for the Lee et al. (2022) and Dupuis et al. (2025) simulation validation studies.

The `cs200_hazard` folder contains the fault and distributed seismicity ERF for the 2010 NZ NHM.


**Reading Data Files:**
- `.pickle` files: Use `pd.read_pickle()` from pandas
- `.parquet` files: Use `pd.read_parquet()` from pandas
- `.npy` files: Use `np.load()` from numpy
- `.pt` files: Use `torch.load(filepath, weights_only=False)` from PyTorch

## Model Training
Use the `./train_models.sh` script for training the models. Note, that the base CV & full models have to be trained first, as these are needed to subsequently train the location-specific models.
A GPU is highly recommended, it may be possible without, but not tested and most likely extremely slow. Modify the number of processes to suit the system used, more processes results in higher memory usage (both main and GPU).

## Post-processing
Use the `run_cv_post.sh` script to compute mixed-effects regression results for the CV runs. For the full models, use `run_full_post.sh` to compute distributed seismicity hazard results. Both are slow and require large amounts of memory, run times were ~12 hours and ~12-24 hours, respectively, on a system with 16 CPU cores and 64 GB of memory. Number of processes is configured for this system, adjust as needed.

## Generating Paper Figure
Use the `gen_paper_figures.sh` script to compute the paper figures. 

