# utils/data_loader.py

import pandas as pd
import os
from datetime import datetime, timedelta


def load_ohio_patient_data(patient_id, folder):
    """
    Load CGM, insulin, and meal data for a patient from the OhioT1DM dataset.
    :param patient_id: string (e.g. '559')
    :param folder: path to patient data folder (e.g., 'my_data/ohio')
    :return: pd.DataFrame with merged time-aligned data (5-min intervals)
    """
    base_path = os.path.join(folder, patient_id)

    cgm = pd.read_csv(os.path.join(base_path, "CGM.csv"))
    bolus = pd.read_csv(os.path.join(base_path, "InsulinBolus.csv"))
    basal = pd.read_csv(os.path.join(base_path, "InsulinBasal.csv"))
    meals = pd.read_csv(os.path.join(base_path, "Meals.csv"))

    # Convert timestamps
    def parse_time(df):
        return pd.to_datetime(df["timestamp"] if "timestamp" in df else df["datetime"], errors='coerce')

    cgm["time"] = parse_time(cgm)
    bolus["time"] = parse_time(bolus)
    basal["time"] = parse_time(basal)
    meals["time"] = parse_time(meals)

    # Round to 5-min bins
    start = cgm["time"].min().floor("5min")
    end = cgm["time"].max().ceil("5min")
    time_index = pd.date_range(start=start, end=end, freq="5min")
    df = pd.DataFrame(index=time_index)

    df["glucose"] = cgm.set_index("time")["sensor_glucose"].resample("5min").mean()
    df["bolus"] = bolus.set_index("time")["insulin"].resample("5min").sum().fillna(0)
    df["basal"] = basal.set_index("time")["rate"].resample("5min").ffill().fillna(0)
    df["carbs"] = meals.set_index("time")["carbs"].resample("5min").sum().fillna(0)

    df = df.reset_index().rename(columns={"index": "timestamp"})
    df = df.fillna(method="ffill").fillna(0)

    return df


if __name__ == "__main__":
    example = load_ohio_patient_data("559", "my_data/ohio")
    print(example.head())
