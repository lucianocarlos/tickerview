import os
import glob
import json
import pandas as pd
import sys

try:
    import yaml
except ImportError:
    yaml = None

# Adiciona a pasta raiz do processing ao path para importar o motor
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from features_mapper import tratar_nans_series_temporais

def carregar_config_file(caminho_base):
    """Carrega .yaml se existir, senão usa .json como fallback."""
    p_yaml = caminho_base + ".yaml" if not caminho_base.endswith((".yaml", ".yml", ".json")) else caminho_base
    p_json = os.path.splitext(caminho_base)[0] + ".json"
    
    path_final = None
    if os.path.exists(caminho_base) and os.path.isfile(caminho_base):
        path_final = caminho_base
    elif os.path.exists(p_yaml):
        path_final = p_yaml
    elif os.path.exists(p_json):
        path_final = p_json
        
    if not path_final or not os.path.exists(path_final):
        return {}
        
    ext = os.path.splitext(path_final)[1].lower()
    with open(path_final, "r", encoding="utf-8") as f:
        if ext in [".yaml", ".yml"]:
            return yaml.safe_load(f) if yaml else {}
        else:
            return json.load(f)

def processar_precos():
    """
    Pré-processa a base de preços (OHLCV) aplicando as regras do pre_config.yaml / pre_config.json
    através do features_mapper.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

    # --- Dinâmica de Aquisição ---
    raw_base = os.path.join(project_root, "data", "raw")
    config_mestre_dir = os.path.dirname(__file__)
    config_mestre = carregar_config_file(os.path.join(config_mestre_dir, "dataset_config"))
    if not config_mestre:
        config_mestre = carregar_config_file(os.path.join(os.path.dirname(config_mestre_dir), "dataset_config"))
        
    alvo = config_mestre.get("aquisicao_alvo", "latest")
            
    if alvo != "latest" and os.path.exists(os.path.join(raw_base, alvo)):
        aquisicao_atual = alvo
    else:
        aquisicoes = glob.glob(os.path.join(raw_base, "aquisicao_*"))
        aquisicao_atual = os.path.basename(sorted(aquisicoes)[-1]) if aquisicoes else "aquisicao_001"
    aquisicao_dir = os.path.join(raw_base, aquisicao_atual)
    os.makedirs(aquisicao_dir, exist_ok=True)
    input_path = os.path.join(aquisicao_dir, "precos.parquet")
    output_path = os.path.join(project_root, "data", "dataset", "precos_p.parquet")

    config = carregar_config_file(os.path.join(os.path.dirname(__file__), "pre_config"))

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
