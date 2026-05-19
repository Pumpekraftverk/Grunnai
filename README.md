# Predictive maintenance and condition monitoring at Grunnai power plant

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
- `figures/` - Generated figures from the analysis notebooks.

## Clone the Repository

Clone the GitHub repository and move into the project folder:

```bash
git clone https://github.com/Pumpekraftverk/Grunnai.git
cd Grunnai
```

## Create the Conda Environment

Install Conda or Miniconda, and then create the environment from `environment.yml`:

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate signal-env
```

## Run the Notebooks


## What This Repository Does

This project analyzes operational signal data from the Grunnai power plant. It focuses on the relationship between generator/turbine operating conditions and Drive-End (DE) vibration.

The main workflow is:

1. Load signal CSV files and match them with signal metadata from `Grunnåi_signallist.csv`.
2. Clean invalid values.
3. Extract operating periods based on active power.
4. Use PELT change-point detection and fixed-size windows to study changes in operation.
5. Classify steady-state windows using DBSCAN or a power-variation threshold.
6. Build a modelling dataset with features such as power, needle position, exciter current, inlet pressure, and DE vibration.
7. Train and evaluate machine-learning models for vibration prediction.

## Data Notes

Some notebooks expect local data folders that contains .csv files from an external system


## References and Credits

- NumPy: Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). Array programming with NumPy. *Nature*, 585, 357-362. <https://doi.org/10.1038/s41586-020-2649-2>
- pandas: McKinney, W. (2010). Data structures for statistical computing in Python. *Proceedings of the 9th Python in Science Conference*, 51-56. <https://doi.org/10.25080/Majora-92bf1922-00a>
- Matplotlib: Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering*, 9(3), 90-95. <https://doi.org/10.1109/MCSE.2007.55>
- SciPy: Virtanen, P., Gommers, R., Oliphant, T. E., et al. (2020). SciPy 1.0: fundamental algorithms for scientific computing in Python. *Nature Methods*, 17, 261-272. <https://doi.org/10.1038/s41592-019-0686-2>
- scikit-learn: Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830. <https://jmlr.org/papers/v12/pedregosa11a.html>
- ruptures: Truong, C., Oudre, L., & Vayatis, N. (2020). Selective review of offline change point detection methods. *Signal Processing*, 167, 107299. <https://doi.org/10.1016/j.sigpro.2019.107299>
- dcor: Ramos-Carreno, C., & Torrecilla, J. L. (2023). dcor: Distance correlation and energy statistics in Python. *SoftwareX*, 22, 101326. <https://doi.org/10.1016/j.softx.2023.101326>
- ydata-profiling: Sequeira, R., et al. (2023). ydata-profiling: Accelerating data-centric AI with high-quality data. *Neurocomputing*, 554, 126585. <https://doi.org/10.1016/j.neucom.2023.126585>
