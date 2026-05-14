from pathlib import Path

import dcor
import numpy as np
import pandas as pd
import ruptures as rpt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------


SIGNAL_COLUMNS = ["Datetime", "signal"]
SIGNAL_FILE_COLUMNS = ["name", "external_id", "file"]
SIGNAL_DATA_COLUMNS = ["name", "external_id", "signal_df", "unit", "file"]
CLEANING_SUMMARY_COLUMNS = ["Signal", "Invalid values", "Invalid [%]"]
OPERATING_PERIOD_COLUMNS = ["start_time", "end_time", "n_samples"]
WINDOW_SORT_COLUMNS = ["operating_period", "start_time"]
WINDOW_COLS = (
    "operating_period",
    "start_time",
    "end_time",
    "mean_power",
    "std_power",
)
RESULT_COLS = WINDOW_COLS + (
    "cluster",
    "is_steady",
)
WINDOW_FEATURE_COLS = list(WINDOW_COLS) + ["start_idx", "end_idx"]
DBSCAN_EPS_COLUMNS = ["epsilon", "n_clusters", "noise_percent"]
CORRELATION_COLUMNS = ["Signal", "Pearson", "Distance"]
STEADY_INTERVAL_AGG = {
    "start_time": ("start_time", "first"),
    "end_time": ("end_time", "last"),
    "n_windows": ("std_power", "size"),
    "mean_power": ("mean_power", "mean"),
    "mean_std_power": ("std_power", "mean"),
    "min_power": ("mean_power", "min"),
    "max_power": ("mean_power", "max"),
}

DEFAULT_SIGNAL_NAME_MAP = {
    "Scada.GRUN.AGG2.G2.MV.M_P_MW": "Generator active power",
    "Scada.GRUN.AGG2.G2.MV.M_PSP": "Set point",
    "Scada.GRUN.AGG2.TURB2.REG.M_TURT": "Rotational speed",
    "Scada.GRUN.AGG2.G2.LAGER.M_VIBR1": "DE vibration",
    "Scada.GRUN.AGG2.G2.MAGN.M_I": "Exciter current",
    "Scada.GRUN.AGG2.TURB2.PADRAG.M_WCKT_POS": "Total needle opening position",
    "Hydrocord.StandardData.G2_Sjakt_Trykk(=A2=HB1=BPA1)Mean": "Turbine inlet pressure",
    "Scada.GRUN.AGG2.G2.LAGER.M_LAGTMP6": "DE bearing temperature",
    "Scada.GRUN.AGG2.G2.LAGER.M_OLTMP2": "DE bearing oil temperature",
}

UNIT_MAP = {
    "Turbine inlet pressure": "kPa",
    "DE bearing temperature": "°C",
    "DE bearing oil temperature": "°C",
    "DE vibration": "mm/s RMS",
    "Exciter current": "A",
    "Generator active power": "MW",
    "Total needle opening position": "%",
    "Rotational speed": "%",
}

# ---------------------------------------------------------------------
# Internal helper functions
# ---------------------------------------------------------------------


def _clean_text(value):
    if pd.isna(value):
        return None

    return " ".join(str(value).strip().split())


def _rename_signal(signal_df, column_name):
    return signal_df[SIGNAL_COLUMNS].rename(columns={"signal": column_name})


def _value_column(df):
    if "signal" in df.columns:
        return "signal"

    value_columns = [col for col in df.columns if col != "Datetime"]

    if not value_columns:
        raise ValueError("Expected a signal column or one non-Datetime value column.")

    return value_columns[0]


def _period_slice(df, period):
    return df[
        df["Datetime"].between(period["start_time"], period["end_time"])
    ].reset_index(drop=True)


def _read_signal_csv(file, n=None):
    nrows = None if n is None else n + 1
    df = pd.read_csv(
        file,
        header=None,
        skiprows=1,
        usecols=[0, 1],
        nrows=nrows,
    )

    unit = None
    if str(df.iloc[0, 0]).strip().lower() == "unit":
        unit = str(df.iloc[0, 1]).strip()
        df = df.iloc[1:]

    df.columns = SIGNAL_COLUMNS
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce").dt.floor("s")
    df["signal"] = pd.to_numeric(df["signal"], errors="coerce")

    return df.reset_index(drop=True), unit


def _find_invalid_values(signal_df, signal_name, vibration_max_value=None):
    signal_name = signal_name.lower()
    values = signal_df["signal"]

    is_temperature_signal = "temp" in signal_name
    is_vibration_signal = "vibrasjon" in signal_name or "vibration" in signal_name

    if is_temperature_signal:
        return values <= 0

    if is_vibration_signal:
        invalid_values = values < 0

        if vibration_max_value is not None:
            invalid_values |= values > vibration_max_value

        return invalid_values

    return values < 0


def _sort_windows(windows):
    return windows.sort_values(WINDOW_SORT_COLUMNS).reset_index(drop=True).copy()


def _add_interval_id(df, change_column):
    df = df.copy()
    df["interval_id"] = (
        (df["operating_period"] != df["operating_period"].shift())
        | (df[change_column] != df[change_column].shift())
    ).cumsum()
    return df


def _steady_intervals(windows, group_columns):
    return (
        windows.loc[windows["is_steady"]]
        .groupby(group_columns, observed=True)
        .agg(**STEADY_INTERVAL_AGG)
        .reset_index()
    )


def _scaled_window_features(windows):
    features = (
        windows[["mean_power", "std_power"]]
        .dropna()
        .astype("float32")
    )

    if features.empty:
        return features, features

    scaled = StandardScaler().fit_transform(features).astype("float32")
    return features, scaled


def _dbscan_labels(features, eps, min_samples):
    return DBSCAN(
        eps=eps,
        min_samples=min_samples,
        algorithm="kd_tree",
        n_jobs=1,
    ).fit_predict(features)


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------


def find_signal_files(
    search_folder,
    signal_file="Grunnåi_signallist.csv",
    name_map=None,
):
    """
    Find CSV files that correspond to known signal external IDs.

    Parameters
    ----------
    search_folder : str or pathlib.Path
        Folder containing exported signal CSV files.
    signal_file : str or pathlib.Path, optional
        CSV file containing signal metadata with external IDs and names.
    name_map : dict, optional
        Optional mapping used to rename selected external IDs.

    Returns
    -------
    pandas.DataFrame
        Table with signal name, external ID, and file path.
    """
        
    signals = pd.read_csv(signal_file, usecols=["CogniteExternalId", "Name"])
    signals["CogniteExternalId"] = (
        signals["CogniteExternalId"].dropna().astype(str).str.strip()
    )
    signals["Name"] = signals["Name"].map(_clean_text)
    signal_names = dict(zip(signals["CogniteExternalId"], signals["Name"]))
    name_map = DEFAULT_SIGNAL_NAME_MAP | (name_map or {})
    rows = []

    for file in sorted(Path(search_folder).glob("*.csv")):
        external_id = file.stem.strip()

        if external_id in signal_names:
            rows.append({
                "name": name_map.get(external_id, signal_names[external_id]),
                "external_id": external_id,
                "file": str(file),
            })

    return pd.DataFrame(rows, columns=SIGNAL_FILE_COLUMNS)


def load_signal_data(
    search_folder,
    n=None,
    signal_file="Grunnåi_signallist.csv",
):
    """
    Load selected signal CSV files into a signal table.

    Each row in the returned table contains the signal name, external ID,
    signal dataframe, unit, and source file.

    Parameters
    ----------
    search_folder : str or pathlib.Path
        Folder containing signal CSV files.
    n : int, optional
        Number of data rows to read from each file. If None, all rows are read.
    signal_file : str or pathlib.Path, optional
        CSV file containing signal metadata.

    Returns
    -------
    pandas.DataFrame
        Signal table with one row per loaded signal.
    """
    
    file_table = find_signal_files(search_folder, signal_file)
    rows = []

    for _, row in file_table.iterrows():
        name = str(row["name"]).strip()
        df, csv_unit = _read_signal_csv(row["file"], n=n)

        rows.append({
            "name": name,
            "external_id": row["external_id"],
            "signal_df": df,
            "unit": UNIT_MAP.get(name, csv_unit),
            "file": row["file"],
        })

    return pd.DataFrame(rows, columns=SIGNAL_DATA_COLUMNS)

# ---------------------------------------------------------------------
# Cleaning and preprocessing
# ---------------------------------------------------------------------


def clean_signals(signal_table, vibration_max_value=None):
    signal_table = signal_table.copy()
    summary_rows = []

    for i, row in signal_table.iterrows():
        signal_name = row["name"]
        signal_df = row["signal_df"].copy()

        invalid_values = _find_invalid_values(
            signal_df, 
            signal_name,
            vibration_max_value=vibration_max_value
        )

        if invalid_values.any():
            summary_rows.append({
                "Signal": signal_name,
                "Invalid values": invalid_values.sum(),
                "Invalid [%]": round(100 * invalid_values.mean(), 4)
            })

            signal_df.loc[invalid_values, "signal"] = np.nan
            signal_df["signal"] = signal_df["signal"].interpolate()

            signal_table.at[i, "signal_df"] = signal_df

    return signal_table, pd.DataFrame(summary_rows, columns=CLEANING_SUMMARY_COLUMNS)


def get_signal_df(signal_table, signal_name):
    matches = signal_table.loc[signal_table["name"] == signal_name, "signal_df"]

    if matches.empty:
        available = "\n".join(sorted(signal_table["name"].astype(str).unique()))
        raise KeyError(
            f"Could not find signal named {signal_name!r}. "
            f"Available signal names are:\n{available}"
        )

    return matches.iloc[0]


# ---------------------------------------------------------------------
# Operating-period detection
# ---------------------------------------------------------------------

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

    return pd.DataFrame(periods, columns=OPERATING_PERIOD_COLUMNS)


# ---------------------------------------------------------------------
# Steady-state detection
# ---------------------------------------------------------------------


def find_pelt_change_points(
    ref_df,
    operating_periods,
    min_size=6,
    jump=1,
):
    value_column = _value_column(ref_df)
    pelt_times = []

    for _, period in operating_periods.iterrows():
        period_df = _period_slice(ref_df, period)
        y = period_df[value_column].to_numpy()

        if len(y) < min_size:
            continue

        penalty = 2 * np.log(len(y))

        bkps = rpt.Pelt(
            model="l2",
            min_size=min_size,
            jump=jump
        ).fit(y).predict(pen=penalty)

        for b in bkps[:-1]:
            pelt_times.append(period_df["Datetime"].iloc[b - 1])

    return pelt_times


def make_window_features(
    ref_df,
    operating_periods,
    window_size=6,
    n_periods=None,
):
    value_column = _value_column(ref_df)
    rows = []

    periods = (
        operating_periods.head(n_periods)
        if n_periods is not None
        else operating_periods
    )

    for period_id, period in periods.iterrows():
        period_df = _period_slice(ref_df, period)

        for start in range(0, len(period_df) - window_size + 1, window_size):
            window = period_df.iloc[start:start + window_size]

            rows.append({
                "operating_period": period_id,
                "start_time": window["Datetime"].iloc[0],
                "end_time": window["Datetime"].iloc[-1],
                "mean_power": window[value_column].mean(),
                "std_power": window[value_column].std(ddof=0),
                "start_idx": start,
                "end_idx": start + window_size - 1
            })

    return pd.DataFrame(rows, columns=WINDOW_FEATURE_COLS)


def test_dbscan_eps(
    windows,
    eps_start=0.05,
    eps_stop=0.40,
    eps_step=0.02,
    min_samples=5
):
    features, scaled_features = _scaled_window_features(windows)

    if features.empty:
        return pd.DataFrame(columns=DBSCAN_EPS_COLUMNS)

    results = []

    for eps in np.arange(eps_start, eps_stop + eps_step, eps_step):
        labels = _dbscan_labels(
            scaled_features,
            eps=eps,
            min_samples=min_samples,
        )

        results.append({
            "epsilon": round(eps, 2),
            "n_clusters": len(set(labels) - {-1}),
            "noise_percent": (labels == -1).mean() * 100
        })

    return pd.DataFrame(results, columns=DBSCAN_EPS_COLUMNS)


def apply_dbscan_to_windows(windows, eps=0.09, min_samples=5):
    data = (
        windows[list(WINDOW_COLS)]
        .dropna(subset=["mean_power", "std_power"])
        .sort_values(WINDOW_SORT_COLUMNS)
        .copy()
    )

    if data.empty:
        data = windows[list(WINDOW_COLS)].copy()
        data["cluster"] = np.nan
        data["is_steady"] = False
        return data

    _, scaled_features = _scaled_window_features(data)
    labels = _dbscan_labels(scaled_features, eps=eps, min_samples=min_samples)
    data["cluster"] = labels
    data["is_steady"] = data["cluster"] != -1

    return data


def extract_steady_states_from_windows(windows, eps=0.09, min_samples=5):
    clustered_windows = apply_dbscan_to_windows(
        windows,
        eps=eps,
        min_samples=min_samples
    )

    clustered_windows = (
        clustered_windows[list(RESULT_COLS)]
        .sort_values(WINDOW_SORT_COLUMNS)
        .reset_index(drop=True)
    )
    clustered_windows = _add_interval_id(clustered_windows, "cluster")
    steady_intervals = _steady_intervals(
        clustered_windows,
        ["operating_period", "interval_id", "cluster"],
    )

    return clustered_windows, steady_intervals


def extract_steady_states(windows, eps=0.09, min_samples=5):
    return extract_steady_states_from_windows(
        windows,
        eps=eps,
        min_samples=min_samples,
    )


def extract_steady_states_by_threshold(windows, std_threshold):
    windows = _sort_windows(windows)
    windows["is_steady"] = windows["std_power"] <= std_threshold
    windows = _add_interval_id(windows, "is_steady")
    steady_intervals = _steady_intervals(
        windows,
        ["operating_period", "interval_id"],
    )

    return windows, steady_intervals


# ---------------------------------------------------------------------
# Feature dataset construction
# ---------------------------------------------------------------------


def window_mean(signal_df, windows):
    signal = signal_df[["Datetime", "signal"]].dropna().sort_values("Datetime")
    intervals = pd.IntervalIndex.from_arrays(
        windows["start_time"],
        windows["end_time"],
        closed="both",
    )

    window_idx = intervals.get_indexer(signal["Datetime"])
    signal = signal.loc[window_idx >= 0].copy()
    signal["window_idx"] = window_idx[window_idx >= 0]

    means = signal.groupby("window_idx")["signal"].mean()
    return pd.Series(
        means.reindex(range(len(windows))).to_numpy(),
        index=windows.index,
    )


def build_modelling_dataset(
    windows,
    needle_position_df,
    exciter_current_df,
    inlet_pressure_df,
    vibration_df,
):
    model_df = windows[
        [
            "operating_period",
            "start_time",
            "end_time",
            "is_steady",
            "interval_id",
            "mean_power",
        ]
    ].rename(columns={
        "mean_power": "power",
    }).copy()

    model_df["needle_position"] = window_mean(needle_position_df, model_df)
    model_df["exciter_current"] = window_mean(exciter_current_df, model_df)
    model_df["inlet_pressure"] = window_mean(inlet_pressure_df, model_df)
    model_df["de_vibration"] = window_mean(vibration_df, model_df)

    value_cols = [
        "power",
        "exciter_current",
        "inlet_pressure",
        "needle_position",
        "de_vibration",
    ]

    model_df.loc[~model_df["is_steady"], value_cols] = np.nan

    return model_df.sort_values("start_time").reset_index(drop=True)


# ---------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------

def get_interval_mask(df, intervals):
    mask = pd.Series(False, index=df.index)

    for _, interval in intervals.iterrows():
        mask |= df["Datetime"].between(
            interval["start_time"],
            interval["end_time"]
        )

    return mask


def correlate_with_vibration(signal_table, operating_periods, vib_name):
    vib = get_signal_df(signal_table, vib_name)

    operating_mask = get_interval_mask(vib, operating_periods)

    vib_operating = vib.loc[
        operating_mask,
        ["Datetime", "signal"]
    ].rename(columns={"signal": "vibration"})

    results = []

    for _, row in signal_table.iterrows():
        name = row["name"]

        if name == vib_name:
            continue

        data = (
            vib_operating
            .merge(_rename_signal(row["signal_df"], name), on="Datetime")
            .dropna()
        )

        if len(data) < 2:
            continue

        results.append({
            "Signal": name,
            "Pearson": data["vibration"].corr(data[name], method="pearson"),
            "Distance": dcor.distance_correlation(
                data["vibration"],
                data[name],
            ),
        })

    return (
        pd.DataFrame(results, columns=CORRELATION_COLUMNS)
        .sort_values("Distance", ascending=False)
        .reset_index(drop=True)
    )
