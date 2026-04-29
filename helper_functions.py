# Basic libraries
import numpy as np
import pandas as pd

# Access folders and extract filenames
import os
from glob import glob

# Scaling and model
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

# PELT algorithm
import ruptures as rpt

# Statistics, correlation
from scipy.stats import pearsonr
import dcor

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

def clean_signals(dfs):
    dfs = dfs.copy()
    rows = []

    for i, row in dfs.iterrows():
        name = row["name"]
        df = row["signal_df"].copy()

        bad = df["signal"] < 0

        if "temp" in name.lower():
            bad |= df["signal"] == 0

        if bad.any():
            rows.append({
                "Signal": name,
                "Invalid values": bad.sum(),
                "Invalid [%]": round(100 * bad.mean(), 4)
            })

            df.loc[bad, "signal"] = np.nan
            df["signal"] = df["signal"].interpolate()

            dfs.at[i, "signal_df"] = df

    return dfs, pd.DataFrame(rows)

def apply_dbscan_to_windows(windows, eps=0.09, min_samples=5):
    windows = windows.copy()

    X = windows[["mean_power", "std_power"]].dropna()

    if X.empty:
        windows["cluster"] = np.nan
        windows["is_steady"] = False
        return windows

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    labels = DBSCAN(
        eps=eps,
        min_samples=min_samples
    ).fit_predict(X_scaled)

    windows = windows.loc[X.index].copy()
    windows["cluster"] = labels
    windows["is_steady"] = windows["cluster"] != -1

    return windows

def extract_steady_states_from_windows(windows, eps=0.09, min_samples=5):
    windows = apply_dbscan_to_windows(
        windows,
        eps=eps,
        min_samples=min_samples
    )

    windows = windows.sort_values(
        ["operating_period", "start_time"]
    ).reset_index(drop=True)

    windows["interval_id"] = (
        (windows["operating_period"] != windows["operating_period"].shift()) |
        (windows["cluster"] != windows["cluster"].shift())
    ).cumsum()

    steady_intervals = (
        windows[windows["is_steady"]]
        .groupby(["operating_period", "interval_id", "cluster"])
        .agg(
            start_time=("start_time", "first"),
            end_time=("end_time", "last"),
            n_windows=("cluster", "size"),
            mean_power=("mean_power", "mean"),
            mean_std_power=("std_power", "mean"),
            min_power=("mean_power", "min"),
            max_power=("mean_power", "max")
        )
        .reset_index()
    )

    return windows, steady_intervals


def extract_operating_periods(
    speed_df,
    power_df,
    speed_threshold=90,
    power_threshold=0.5,
    min_samples=30
):
    speed = speed_df.rename(columns={"signal": "speed"})
    power = power_df.rename(columns={"signal": "power"})

    df = pd.merge(
        power[["Datetime", "power"]],
        speed[["Datetime", "speed"]],
        on="Datetime"
    )

    df["is_operating"] = (
        (df["speed"] >= speed_threshold) &
        (df["power"] >= power_threshold)
    )

    df["period_id"] = (
        df["is_operating"] != df["is_operating"].shift()
    ).cumsum()

    periods = []

    for _, group in df.groupby("period_id"):
        if group["is_operating"].iloc[0] and len(group) >= min_samples:
            periods.append({
                "start_time": group["Datetime"].iloc[0],
                "end_time": group["Datetime"].iloc[-1],
                "n_samples": len(group)
            })

    return pd.DataFrame(periods)


def find_pelt_change_points(power_df, operating_periods, min_samples=30, n_periods=None):
    power = power_df.rename(columns={"signal": "power"})
    pelt_times = []

    periods = operating_periods if n_periods is None else operating_periods.head(n_periods)

    for _, period in periods.iterrows():
        wp = power[
            (power["Datetime"] >= period["start_time"]) &
            (power["Datetime"] <= period["end_time"])
        ]

        y = wp["power"].to_numpy()

        if len(y) < min_samples:
            continue

        penalty = 2 * np.log(len(y))

        bkps = rpt.Pelt(
            model="l2",
            min_size=min_samples
        ).fit(y).predict(pen=penalty)

        for b in bkps[:-1]:
            pelt_times.append(wp["Datetime"].iloc[b - 1])

    return pelt_times

def make_window_features(power_df, operating_periods, window_size=6, n_periods=None):
    power = power_df.rename(columns={"signal": "power"})
    rows = []

    periods = operating_periods.head(n_periods) if n_periods else operating_periods

    for period_id, period in periods.iterrows():
        wp = power[
            power["Datetime"].between(period["start_time"], period["end_time"])
        ].reset_index(drop=True)

        for start in range(0, len(wp) - window_size + 1, window_size):
            window = wp.iloc[start:start + window_size]

            rows.append({
                "operating_period": period_id,
                "start_time": window["Datetime"].iloc[0],
                "end_time": window["Datetime"].iloc[-1],
                "mean_power": window["power"].mean(),
                "std_power": window["power"].std(ddof=0),
                "start_idx": start,
                "end_idx": start + window_size - 1
            })

    return pd.DataFrame(rows)

def test_dbscan_eps(
    windows,
    eps_start=0.05,
    eps_stop=0.40,
    eps_step=0.02,
    min_samples=5
):
    X = windows[["mean_power", "std_power"]].dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = []

    for eps in np.arange(eps_start, eps_stop + eps_step, eps_step):
        labels = DBSCAN(
            eps=eps,
            min_samples=min_samples
        ).fit_predict(X_scaled)

        n_clusters = len(set(labels) - {-1})
        noise_percent = (labels == -1).mean() * 100

        results.append({
            "epsilon": round(eps, 2),
            "n_clusters": n_clusters,
            "noise_percent": noise_percent
        })

    return pd.DataFrame(results)



def correlate_with_vibration(dfs, power_df, operating_periods, vib_name):
    vib = dfs.loc[dfs["name"] == vib_name, "signal_df"].iloc[0]

    # Get timestamps during operating periods
    operating_times = []
    for _, period in operating_periods.iterrows():
        mask = power_df["Datetime"].between(period["start_time"], period["end_time"])
        operating_times.append(power_df.loc[mask, ["Datetime"]])

    operating_times = pd.concat(operating_times).drop_duplicates()

    # Keep vibration only during operating periods
    vib_operating = operating_times.merge(vib, on="Datetime")
    vib_operating = vib_operating.rename(columns={"signal": "vibration"})

    results = []

    for _, row in dfs.iterrows():
        name = row["name"]

        if name == vib_name:
            continue

        data = vib_operating.merge(row["signal_df"], on="Datetime")
        data = data.rename(columns={"signal": name})
        data = data.dropna()

        if len(data) < 2:
            continue

        results.append({
            "Signal": name,
            "Pearson": pearsonr(data["vibration"], data[name])[0],
            "Distance": dcor.distance_correlation(data["vibration"], data[name])
        })

    return pd.DataFrame(results).sort_values("Distance", ascending=False)