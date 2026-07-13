import os
import glob
import json
import pandas as pd

def processar_cvm():
    """
    Pré-processa a base contábil da CVM aplicando as regras paramétricas e 
    traduzindo os códigos numéricos para rótulos em português.
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
    input_path = os.path.join(aquisicao_dir, "cvm_dfp.parquet")
    output_path = os.path.join(project_root, "data", "dataset", "cvm_dfp_p.parquet")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    if not os.path.exists(input_path):
        return None

    df = pd.read_parquet(input_path)

    # Dicionário de Tradução CVM
    # Substituindo os códigos por nomenclaturas que humanos (e classificadores) entendem.
    mapeamento_cvm = {
        '1': 'Ativo_Total',
        '1.01.01': 'Caixa_Equivalentes',
        '2.01.04': 'Emprestimos_Curto_Prazo',
        '2.02.01': 'Emprestimos_Longo_Prazo',
        '2.03': 'Patrimonio_Liquido',
        '3.05': 'EBIT_Operacional',
        '3.11': 'Lucro_Prejuizo_Periodo'
    }
    
    df = df.rename(columns=mapeamento_cvm)

    # A imputação de NaNs (zeros ou medianas) foi delegada para o MLOps
    # para permitir o Grid Search sem Data Leakage.

    # Engenharia de Features Internas (Ratios Contábeis base)
    if 'Emprestimos_Curto_Prazo' in df.columns and 'Emprestimos_Longo_Prazo' in df.columns:
        df['Divida_Total'] = df['Emprestimos_Curto_Prazo'] + df['Emprestimos_Longo_Prazo']

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    return df

if __name__ == "__main__":
    processar_cvm()
