import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def tratar_nans_series_temporais(df, colunas, estrategia):
    """
    Aplica estratégias de NaN para séries temporais (OHLCV).
    Essa é a ÚNICA função que sobreviveu na camada de extração (processing).
    As imputações populacionais e tratamento de outliers/scaling foram movidas para o classificador MLOps.
    """
    if estrategia == "drop":
        df = df.dropna(subset=colunas)
    elif estrategia == "ffill":
        df[colunas] = df[colunas].ffill()
    elif estrategia == "bfill":
        df[colunas] = df[colunas].bfill()
    elif estrategia == "interpolate_linear":
        df[colunas] = df[colunas].interpolate(method="linear")
    return df

