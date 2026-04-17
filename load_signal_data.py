import os
from glob import glob
import pandas as pd

def load_signal_data(n, search_folder, signal_file="Grunnåi_signallist.csv"):
    """
    Load and process signal CSV files based on a signal list.

    Parameters
    ----------
    n : int
        Number of rows to read from each signal file.
    search_folder : str
        Folder containing the measurement CSV files.
    signal_file : str, optional
        Path to the signal list CSV file.
        Default is "Grunnåi_signallist.csv".

    Returns
    -------
    dfs : list[pd.DataFrame]
        List of processed DataFrames, one for each signal.
    files : dict
        Dictionary mapping signal name -> file path.
    """

    # Read the signal list and create mapping:
    # CogniteExternalId -> signal name
    signals = pd.read_csv(signal_file)
    name_map = dict(
        zip(
            signals["CogniteExternalId"].astype(str).str.strip(),
            signals["Name"].astype(str).str.strip()
        )
    )

    # Store matching files
    files = {}

    # Search for CSV files and map them to signal names
    for file in glob(os.path.join(search_folder, "*.csv")):
        cognite_id = os.path.splitext(os.path.basename(file))[0].strip()
        if cognite_id in name_map:
            files[name_map[cognite_id]] = file

    # Load and process each file
    dfs = []

    for name, path in files.items():
        df = pd.read_csv(path, header=None, nrows=n+1, skiprows=1, usecols=[0, 1])

        if not df.empty and str(df.iloc[0, 0]).strip().lower() == "unit":
            df = df.iloc[1:]

        df.columns = ["Timestamp", name]

        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce").dt.floor("s")
        df[name] = pd.to_numeric(df[name], errors="coerce")

        dfs.append(df)

    return dfs, files