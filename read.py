import pandas as pd

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