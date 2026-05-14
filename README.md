# Grunnai EDA

Exploratory data analysis and modelling workflow for Grunnai hydropower signal data.

The repository contains notebooks and helper functions for loading time-series signal exports, cleaning invalid measurements, identifying operating periods, detecting steady-state windows, producing plots, and building a modelling dataset for DE vibration analysis.

## Repository Contents

- `environment.yml` - Conda environment definition for the project.
- `helper_functions.py` - Shared functions for loading data, cleaning signals, detecting operating periods, extracting steady states, and building modelling datasets.
- `EDA.ipynb` - General exploratory analysis, profiling, signal cleaning, and early visualizations.
- `plotting.ipynb` and `plot_raw.ipynb` - Plotting notebooks for raw and processed signals.
- `PELT.ipynb` - Change-point detection using the PELT algorithm.
- `FeatureExtraction.ipynb` - Window feature extraction, steady-state detection, and modelling dataset creation.
- `modeling.ipynb` - Model training and evaluation using the extracted steady-state dataset.
- `profiles/` and `signal_data_report.html` - Generated profiling reports.
- `*.png` - Generated figures from the analysis notebooks.

## Clone the Repository

Clone the GitHub repository and move into the project folder:

```bash
git clone https://github.com/Pumpekraftverk/Grunnai.git
cd Grunnai
```

## Create the Conda Environment

Install Conda or Miniconda if it is not already installed. Then create the environment from `environment.yml`:

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate signal-env
```

If the environment already exists and `environment.yml` has changed, update it with:

```bash
conda env update -f environment.yml --prune
```

The `--prune` flag removes packages that are no longer listed in the YAML file.

## Run the Notebooks

Start Jupyter from the activated environment:

```bash
jupyter notebook
```

Then open the notebooks in the browser. A typical workflow is:

1. `EDA.ipynb` - inspect and clean the signal data.
2. `PELT.ipynb` - detect change points in operating periods.
3. `FeatureExtraction.ipynb` - create window features and steady-state intervals.
4. `modeling.ipynb` - train and evaluate models using the generated dataset.

## What This Repository Does

This project analyzes operational signal data from Grunnai. It focuses on the relationship between generator/turbine operating conditions and DE vibration.

The main workflow is:

1. Load signal CSV files and match them with signal metadata from `Grunnåi_signallist.csv`.
2. Clean invalid values, such as negative values where they are not physically meaningful.
3. Extract operating periods based on active power.
4. Use PELT change-point detection and fixed-size windows to study changes in operation.
5. Classify steady-state windows using DBSCAN or a power-variation threshold.
6. Build a modelling dataset with features such as power, needle position, exciter current, inlet pressure, and DE vibration.
7. Train and evaluate machine-learning models for vibration prediction.

## Data Notes

Some notebooks expect local data folders such as:

- `RawMeasurements/`
- `Meas1minInterpolated/`
- `step_interpolation_10s/`
- `2024-01-01_2024-12-31_step_interpolation_10s/`

Make sure these folders are present before running the notebooks that depend on them.

## Environment Packages

The current Conda environment includes the packages used by the code and notebooks:

- `numpy`
- `pandas`
- `matplotlib`
- `scipy`
- `scikit-learn`
- `ruptures`
- `dcor`
- `ydata-profiling`

