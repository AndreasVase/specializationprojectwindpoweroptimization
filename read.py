import pandas as pd
import pyarrow
import fastparquet

def load_parameters_from_csv(path):
    """
    Leser parameters.csv med pandas og returnerer seks lister:
    CM_up, CM_down, DA, EAM_up, EAM_down, wind_speed
    """
    df = pd.read_csv(path)

    CM_up      = df["CM_up"].tolist()
    CM_down    = df["CM_down"].tolist()
    DA         = df["DA"].tolist()
    EAM_up     = df["EAM_up"].tolist()
    EAM_down   = df["EAM_down"].tolist()
    wind_speed = df["wind_speed"].tolist()

    return CM_up, CM_down, DA, EAM_up, EAM_down, wind_speed



def load_expected_values_from_csv(path):
    """
    Leser parameters.csv med pandas og returnerer forventede verdier:
    P_CM_up, P_CM_down, P_DA, P_EAM_up, P_EAM_down, Q_mean
    """
    df = pd.read_csv(path)

    # Forventede (gjennomsnittlige) priser og vind
    P_CM_up    = df["CM_up"].mean()
    P_CM_down  = df["CM_down"].mean()
    P_DA       = df["DA"].mean()
    P_EAM_up   = df["EAM_up"].mean()
    P_EAM_down = df["EAM_down"].mean()
    Q_mean     = df["wind_speed"].mean()   # tilgjengelig produksjonskapasitet

    return P_CM_up, P_CM_down, P_DA, P_EAM_up, P_EAM_down, Q_mean


path = "mmo_datasets/NO3/roan/production_forecasts.parquet"

def load_mmo_data():
    df = pd.read_parquet(path)

    print(df.head(100))



load_mmo_data()