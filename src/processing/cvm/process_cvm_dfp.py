import os
import glob
import json
import pandas as pd

try:
    import yaml
except ImportError:
    yaml = None

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

def processar_cvm():
    """
    Pré-processa a base contábil da CVM aplicando as regras paramétricas e 
    traduzindo os códigos numéricos para rótulos em português.
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
    input_path = os.path.join(aquisicao_dir, "cvm_dfp.parquet")
    output_path = os.path.join(project_root, "data", "dataset", "cvm_dfp_p.parquet")

    config = carregar_config_file(os.path.join(os.path.dirname(__file__), "pre_config"))

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
