import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

file_mapping = {
    "dayahead_forecasts.parquet": "dayahead_forecasts",
    "dayahead_prices.parquet": "dayahead_prices",
    "imbalance_forecasts.parquet": "imbalance_forecasts",
    "imbalance_prices.parquet": "imbalance_prices",
    "mfrr_cm_down_forecasts.parquet": "mfrr_cm_down_forecasts",
    "mfrr_cm_down_prices.parquet": "mfrr_cm_down_prices",
    "mfrr_cm_up_forecasts.parquet": "mfrr_cm_up_forecasts",
    "mfrr_cm_up_prices.parquet": "mfrr_cm_up_prices",
    "mfrr_eam_down_forecasts.parquet": "mfrr_eam_down_forecasts",
    "mfrr_eam_down_prices.parquet": "mfrr_eam_down_prices",
    "mfrr_eam_up_forecasts.parquet": "mfrr_eam_up_forecasts",
    "mfrr_eam_up_prices.parquet": "mfrr_eam_up_prices",
    "production_forecasts.parquet": "production_forecasts",
    "production.parquet": "production",
}

# scenario columns 0..49
SCENARIO_COLS = [str(i) for i in range(50)]

# Forecast files that must have all-positive scenarios
SCENARIO_FILES = [
    "dayahead_forecasts.parquet",
    "mfrr_cm_down_forecasts.parquet",
    "mfrr_cm_up_forecasts.parquet",
    "mfrr_eam_up_forecasts.parquet",
]


def keep_earliest_per_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (prediction_for, area/park/frequency), keep the earliest created_at row.
    If created_at is missing, return df unchanged.
    """
    if "created_at" not in df.columns:
        return df

    # Make sure created_at is sortable (works if it's int ms or string datetime)
    df = df.copy()
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")

    df_sorted = df.sort_values("created_at")

    # Use prediction_for + area/park/frequency as identity
    subset_cols = ["prediction_for"]
    for col in ["area", "park", "frequency"]:
        if col in df.columns:
            subset_cols.append(col)

    df_first = df_sorted.drop_duplicates(subset=subset_cols, keep="first")
    return df_first


def get_valid_times() -> set:
    """
    Return set of prediction_for timestamps where, for each of the
    four products (dayahead, mfrr_cm_down, mfrr_cm_up, mfrr_eam_up),
    the *earliest* forecast for that prediction_for has all scenarios 0..49 > 0.
    """
    valid_times = None

    for fname in SCENARIO_FILES:
        path = RAW_DIR / fname
        df = pd.read_parquet(path)

        # Only the earliest version per prediction_for (+ area/park/frequency)
        df = keep_earliest_per_prediction(df)

        scenario_cols = [c for c in SCENARIO_COLS if c in df.columns]
        if len(scenario_cols) < 50:
            raise ValueError(
                f"{fname}: expected 50 scenario cols '0'..'49', "
                f"found {len(scenario_cols)} ({scenario_cols})"
            )

        # All scenarios > 0
        mask_pos = df[scenario_cols].gt(0).all(axis=1)

        times_this_file = set(df.loc[mask_pos, "prediction_for"])

        if valid_times is None:
            valid_times = times_this_file
        else:
            valid_times &= times_this_file

    return valid_times or set()


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    valid_times = get_valid_times()
    print("Number of valid time slots (raw):", len(valid_times))

    # Optional human-readable debug
    try:
        valid_times_dt = pd.to_datetime(list(valid_times), utc=True, errors="coerce")
        valid_times_dt = valid_times_dt.sort_values()
        print("First few valid times:", valid_times_dt[:5].to_list())
    except Exception as e:
        print("Could not convert valid_times to datetime for printing:", e)

    for fname in file_mapping.keys():
        in_path = RAW_DIR / fname
        out_path = PROCESSED_DIR / fname

        df = pd.read_parquet(in_path)
        original_len = len(df)

        # --- case 1: forecast files that must obey scenario positivity ---
        if fname in SCENARIO_FILES:
            # Only earliest version per prediction_for (+ area/park/frequency)
            df = keep_earliest_per_prediction(df)

            # Keep only valid time slots
            df = df[df["prediction_for"].isin(valid_times)]

            # Enforce scenario positivity again at row-level
            scenario_cols = [c for c in SCENARIO_COLS if c in df.columns]
            if scenario_cols:
                pos_mask = df[scenario_cols].gt(0).all(axis=1)
                df = df[pos_mask]

                # sanity check: no negatives
                assert (df[scenario_cols] > 0).all().all(), \
                    f"Negative scenario values slipped through in {fname}"

        # --- NEW: mfrr_eam_down_forecasts.parquet -> earliest version per timestamp ---
        elif fname == "mfrr_eam_down_forecasts.parquet":
            # Only earliest version per prediction_for (+ area/park/frequency)
            df = keep_earliest_per_prediction(df)

            # And restrict to the same valid timestamps as the others
            if "prediction_for" in df.columns:
                df = df[df["prediction_for"].isin(valid_times)]

        # --- case 2: other forecast-style files with prediction_for ---
        elif "prediction_for" in df.columns:
            df = df[df["prediction_for"].isin(valid_times)]

        # --- case 3: realized data with 'time' ---
        elif "time" in df.columns:
            df = df[df["time"].isin(valid_times)]

        # else: leave df unchanged (or decide what you want)

        df.to_parquet(out_path, index=False)
        print(f"{fname}: {original_len} -> {len(df)} rows written to {out_path}")



if __name__ == "__main__":
    main()
