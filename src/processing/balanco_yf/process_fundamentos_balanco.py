import os
import glob
import json
import pandas as pd

def processar_balanco_yf():
    """
    Prepara o balanço do yfinance como features complementares ao CVM.
    Lida com o alto nível de NaNs testando diferentes estratégias.
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
    input_path = os.path.join(aquisicao_dir, "fundamentos_balanco.parquet")
    output_path = os.path.join(project_root, "data", "dataset", "fundamentos_balanco_p.parquet")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if not os.path.exists(input_path):
        return None

    df = pd.read_parquet(input_path)

    # Pivotar de formato Longo para Largo (uma coluna por métrica contábil do yf)
    df_pivoted = df.pivot(index=['ticker', 'Data_Referencia'], columns='Metrica', values='Valor').reset_index()

    # A decisão de preencher os nulos estruturais deste balanço (com zero ou medianas)
    # agora é delegada exclusivamente ao MLOps (via Grid Search).
    
    # Renomeando Data_Referencia para facilitar JOIN posterior
    df_pivoted = df_pivoted.rename(columns={'Data_Referencia': 'data'})

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_pivoted.to_parquet(output_path, index=False)
    return df_pivoted

if __name__ == "__main__":
    processar_balanco_yf()
