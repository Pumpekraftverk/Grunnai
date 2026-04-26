import os
from glob import glob
import pandas as pd


def find_signal_files(search_folder, signal_file="Grunnåi_signallist.csv"):
    signals = pd.read_csv(signal_file)

    rows = []

    for file in glob(os.path.join(search_folder, "*.csv")):
        cognite_id = os.path.splitext(os.path.basename(file))[0].strip()

        match = signals[
            signals["CogniteExternalId"].astype(str).str.strip() == cognite_id
        ]

        if not match.empty:
            rows.append({
                "name": str(match["Name"].iloc[0]).strip(),
                "file": file
            })

    return pd.DataFrame(rows)


def load_signal_data(search_folder, n=None, signal_file="Grunnåi_signallist.csv"):
    file_table = find_signal_files(search_folder, signal_file)

    rows = []

    for _, row in file_table.iterrows():
        name = str(row["name"]).strip()
        file = row["file"]

        if n is None:
            df = pd.read_csv(file, header=None, skiprows=1, usecols=[0, 1])
        else:
            df = pd.read_csv(file, header=None, skiprows=1, usecols=[0, 1], nrows=n + 1)

        unit = None

        if str(df.iloc[0, 0]).strip().lower() == "unit":
            unit = str(df.iloc[0, 1]).strip()
            df = df.iloc[1:]

        df.columns = ["Datetime", "signal"]
        df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce").dt.floor("s")
        df["signal"] = pd.to_numeric(df["signal"], errors="coerce")
        df = df.reset_index(drop=True)

        rows.append({
            "name": name,
            "signal_df": df,
            "unit": unit,
            "file": file
        })

    return pd.DataFrame(rows)