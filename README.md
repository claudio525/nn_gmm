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
