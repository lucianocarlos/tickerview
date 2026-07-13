import os
import glob
import json
import pandas as pd
import sys

# Adiciona a pasta raiz do processing ao path para importar o motor
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from features_mapper import tratar_nans_series_temporais

def processar_precos():
    """
    Pré-processa a base de preços (OHLCV) aplicando as regras do pre_config.json
    através do features_mapper.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

    # --- Dinâmica de Aquisição ---
    raw_base = os.path.join(project_root, "data", "raw")
    config_mestre_path = os.path.join(os.path.dirname(__file__), "dataset_config.json")
    if not os.path.exists(config_mestre_path):
        config_mestre_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset_config.json")
        
    alvo = "latest"
    if os.path.exists(config_mestre_path):
        with open(config_mestre_path, "r", encoding="utf-8") as f:
            alvo = json.load(f).get("aquisicao_alvo", "latest")
            
    if alvo != "latest" and os.path.exists(os.path.join(raw_base, alvo)):
        aquisicao_atual = alvo
    else:
        aquisicoes = glob.glob(os.path.join(raw_base, "aquisicao_*"))
        aquisicao_atual = os.path.basename(sorted(aquisicoes)[-1]) if aquisicoes else "aquisicao_001"
    aquisicao_dir = os.path.join(raw_base, aquisicao_atual)
    os.makedirs(aquisicao_dir, exist_ok=True)
    config_path = os.path.join(os.path.dirname(__file__), "pre_config.json")
    input_path = os.path.join(aquisicao_dir, "precos.parquet")
    output_path = os.path.join(project_root, "data", "dataset", "precos_p.parquet")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if not os.path.exists(input_path):
        return None

    df = pd.read_parquet(input_path)

    # 1. Tratamento de NaNs baseado em parâmetro
    colunas_ohlcv = ['Open', 'High', 'Low', 'Close', 'Volume']
    strategy_nan = config.get("ohlcv_nan_strategy", "drop")
    df = tratar_nans_series_temporais(df, colunas_ohlcv, strategy_nan)

    # Os tratamentos de Outliers e Scaling foram movidos para o MLOps
    # para evitar Data Leakage Temporal. Aqui mantemos apenas o NaNs.

    # Cálculo de retorno
    # Como os dados estao em formato longo por ticker, precisamos agrupar
    df = df.sort_values(by=['ticker', 'Date'])
    df['Retorno'] = df.groupby('ticker')['Close'].pct_change()

    # Salva na pasta predata
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    return df

if __name__ == "__main__":
    processar_precos()
