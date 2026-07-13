import os
import glob
import json
import pandas as pd

def processar_info():
    """
    Limpa os metadados do yfinance (Beta, Sector, Industry) e resolve NaNs paramétricos.
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
    input_path = os.path.join(aquisicao_dir, "fundamentos_info.parquet")
    output_path = os.path.join(project_root, "data", "dataset", "fundamentos_info_p.parquet")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if not os.path.exists(input_path):
        return None

    df = pd.read_parquet(input_path)

    # O formato atual é Tidy (Ticker, Metrica, Valor). Precisamos pivotar para ter Colunas.
    df_pivoted = df.pivot(index=['ticker', 'Data_Extracao'], columns='Metrica', values='Valor').reset_index()

    # Selecionamos apenas métricas úteis para evitar inflar o classificador com lixo textual (ex: telefones)
    metricas_chave = ['sector', 'industry', 'beta', 'marketCap', 'trailingPE']
    colunas_presentes = ['ticker'] + [col for col in metricas_chave if col in df_pivoted.columns]
    df_pivoted = df_pivoted[colunas_presentes]

    # Converte variáveis que são numéricas (pois estavam como string devido ao pivot)
    for col in ['beta', 'marketCap', 'trailingPE']:
        if col in df_pivoted.columns:
            df_pivoted[col] = pd.to_numeric(df_pivoted[col], errors='coerce')

    # A imputação do Beta (mediana global ou por setor) foi movida para o pipeline do Scikit-Learn
    # para ser aprendida estritamente no Treino (Holdout/TSP).

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_pivoted.to_parquet(output_path, index=False)
    return df_pivoted

if __name__ == "__main__":
    processar_info()
