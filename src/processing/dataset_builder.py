import os
import json
import glob
import sys
import duckdb

# Garante que o Python ache o indicadores na mesma pasta
sys.path.append(os.path.dirname(__file__))

from indicadores import (
    calcular_PL,
    calcular_PVP,
    calcular_ROE,
    calcular_DY,
    calcular_Retorno_Diario,
    calcular_Volatilidade,
    calcular_Momentum,
    calcular_Anomalia_Volume,
    calcular_RSI,
    calcular_MACD,
    calcular_Bollinger_Width,
    calcular_Alavancagem,
    calcular_Margem_EBIT,
    calcular_ZScore_Setorial
)


def construir_tabela_mestre():
    """
    Script orquestrador final do Data Prep.
    Lê os parquets limpos pelos módulos individuais e utiliza DuckDB ASOF JOIN
    para alinhar as variáveis contábeis trimestrais com os preços diários,
    evitando Look-Ahead Bias. Em seguida, calcula métricas derivadas.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    # --- Dinâmica de Aquisição ---
    raw_base = os.path.join(project_root, "data", "raw")
    config_mestre_path = os.path.join(os.path.dirname(__file__), "dataset_config.json")
    if not os.path.exists(config_mestre_path):
        config_mestre_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "dataset_config.json"
        )

    alvo = "latest"
    if os.path.exists(config_mestre_path):
        with open(config_mestre_path, "r", encoding="utf-8") as f:
            alvo = json.load(f).get("aquisicao_alvo", "latest")

    if alvo != "latest" and os.path.exists(os.path.join(raw_base, alvo)):
        aquisicao_atual = alvo
    else:
        aquisicoes = glob.glob(os.path.join(raw_base, "aquisicao_*"))
        aquisicao_atual = (
            os.path.basename(sorted(aquisicoes)[-1]) if aquisicoes else "aquisicao_001"
        )

    predata_dir = os.path.join(project_root, "data", "dataset")

    precos_path = os.path.join(predata_dir, "precos_p.parquet")
    cvm_path = os.path.join(predata_dir, "cvm_dfp_p.parquet")
    info_path = os.path.join(predata_dir, "fundamentos_info_p.parquet")
    balanco_yf_path = os.path.join(predata_dir, "fundamentos_balanco_p.parquet")

    if not os.path.exists(precos_path):
        print(
            f"Erro: {precos_path} não encontrado. Execute process_precos.py primeiro."
        )
        return

    con = duckdb.connect(database=":memory:")

    query = f"""
    SELECT 
        p.*,
        c.Ativo_Total, c.Caixa_Equivalentes, c.Emprestimos_Curto_Prazo, 
        c.Emprestimos_Longo_Prazo, c.Patrimonio_Liquido, c.EBIT_Operacional, 
        c.Lucro_Prejuizo_Periodo, c.Divida_Total,
        i.sector, i.industry, i.beta, i.marketCap, i.trailingPE,
        b.* EXCLUDE (ticker, data)
    FROM read_parquet('{precos_path}') p
    ASOF JOIN read_parquet('{cvm_path}') c 
        ON p.ticker = c.ticker AND p.Date >= c.data_referencia
    LEFT JOIN read_parquet('{info_path}') i 
        ON p.ticker = i.ticker
    ASOF JOIN read_parquet('{balanco_yf_path}') b
        ON p.ticker = b.ticker AND p.Date >= b.data
    ORDER BY p.ticker, p.Date
    """

    try:
        print("Executando ASOF JOIN temporal com DuckDB...")
        df_mestre = con.execute(query).df()
    except Exception as e:
        print(f"Erro ao juntar as tabelas: {e}")
        return

    print(
        f"Tabela unificada gerada com {df_mestre.shape[0]} linhas e {df_mestre.shape[1]} colunas."
    )

    # =========================================================================
    # ENGENHARIA DE FEATURES (Métricas Derivadas Dinâmicas via Calculadora)
    # =========================================================================
    print("Calculando Ratios Financeiros e Séries Temporais dinâmicas...")

    # 1. Retorno Diário e Volatilidade
    if "Close" in df_mestre.columns:
        df_mestre["Retorno_Diario"] = df_mestre.groupby("ticker")["Close"].transform(
            calcular_Retorno_Diario
        )
        df_mestre["Volatilidade_21d"] = df_mestre.groupby("ticker")["Close"].transform(
            lambda x: calcular_Volatilidade(x, janela=21)
        )

    # 2. P/L Dinâmico
    if (
        "marketCap" in df_mestre.columns
        and "Lucro_Prejuizo_Periodo" in df_mestre.columns
    ):
        df_mestre["P_L_Calculado"] = calcular_PL(
            df_mestre["marketCap"], df_mestre["Lucro_Prejuizo_Periodo"]
        )

    # 3. P/VP Dinâmico
    if "marketCap" in df_mestre.columns and "Patrimonio_Liquido" in df_mestre.columns:
        df_mestre["P_VP_Calculado"] = calcular_PVP(
            df_mestre["marketCap"], df_mestre["Patrimonio_Liquido"]
        )

    # 4. ROE Trimestral
    if (
        "Lucro_Prejuizo_Periodo" in df_mestre.columns
        and "Patrimonio_Liquido" in df_mestre.columns
    ):
        df_mestre["ROE_Calculado"] = calcular_ROE(
            df_mestre["Lucro_Prejuizo_Periodo"], df_mestre["Patrimonio_Liquido"]
        )

    # 5. DY Dinâmico
    if "Total_Dividendos" in df_mestre.columns and "marketCap" in df_mestre.columns:
        df_mestre["DY_Calculado"] = calcular_DY(
            df_mestre["Total_Dividendos"], df_mestre["marketCap"]
        )

    # 6. Momentum (1d, 5d, 10d, 21d, 63d)
    if "Close" in df_mestre.columns:
        for dias in [1, 5, 10, 21, 63]:
            df_mestre[f"Momentum_{dias}d"] = df_mestre.groupby("ticker")["Close"].transform(
                lambda x: calcular_Momentum(x, janela=dias)
            )

    # 7. Técnicos (RSI, MACD, Bollinger Width)
    if "Close" in df_mestre.columns:
        df_mestre["RSI_14d"] = df_mestre.groupby("ticker")["Close"].transform(
            lambda x: calcular_RSI(x, janela=14)
        )
        df_mestre["MACD_Hist"] = df_mestre.groupby("ticker")["Close"].transform(
            lambda x: calcular_MACD(x)
        )
        df_mestre["Bollinger_Width_21d"] = df_mestre.groupby("ticker")["Close"].transform(
            lambda x: calcular_Bollinger_Width(x, janela=21)
        )

    # 8. Anomalia de Volume
    if "Volume" in df_mestre.columns:
        df_mestre["Volume_Anomaly_21d"] = df_mestre.groupby("ticker")["Volume"].transform(
            lambda x: calcular_Anomalia_Volume(x, janela=21)
        )

    # 9. Ratios Financeiros Adicionais
    if "Divida_Total" in df_mestre.columns and "Ativo_Total" in df_mestre.columns:
        df_mestre["Alavancagem_Calculada"] = calcular_Alavancagem(
            df_mestre["Divida_Total"], df_mestre["Ativo_Total"]
        )
    if "EBIT_Operacional" in df_mestre.columns and "Total Revenue" in df_mestre.columns:
        df_mestre["Margem_EBIT_Calculada"] = calcular_Margem_EBIT(
            df_mestre["EBIT_Operacional"], df_mestre["Total Revenue"]
        )

    # 10. Z-Score Cross-Sectional Setorial Diário
    # Aplicar em algumas features numéricas importantes (ex: P_L_Calculado, ROE_Calculado)
    if "sector" in df_mestre.columns and "Date" in df_mestre.columns:
        # Só calcula se a feature existe
        cross_sect_cols = ["P_L_Calculado", "P_VP_Calculado", "ROE_Calculado", "Momentum_21d", "Alavancagem_Calculada"]
        for col in cross_sect_cols:
            if col in df_mestre.columns:
                # Agrupamos por data e então por setor (temos que passar a data tbm pro transform)
                # O mais eficiente no pandas: groupby(['Date', 'sector']) e então tira o zscore.
                df_mestre[f"{col}_zscore_setorial"] = df_mestre.groupby(['Date', 'sector'])[col].transform(
                    lambda x: (x - x.mean()) / (x.std() + 1e-8)
                )

    print(
        "O Pipeline de MLOps agora é responsável pela Limpeza Final (Nulos, Outliers, Scaling) para evitar Data Leakage."
    )

    # =========================================================================
    # CATALOGAÇÃO E VERSIONAMENTO (Dataset Registry Isolado)
    # =========================================================================
    import shutil
    from datetime import datetime

    experiments_dir = predata_dir
    os.makedirs(experiments_dir, exist_ok=True)
    registry_path = os.path.join(experiments_dir, "master_registry.json")

    # Descobre o próximo ID
    existentes = glob.glob(os.path.join(experiments_dir, "dataset*"))
    ids = []
    for f in existentes:
        if os.path.isdir(f):
            basename = os.path.basename(f)
            try:
                ids.append(int(basename.replace("dataset", "")))
            except ValueError:
                pass
    next_id = max(ids) + 1 if ids else 1
    version_str = f"dataset{next_id:03d}"

    exp_folder = os.path.join(experiments_dir, version_str)
    os.makedirs(exp_folder, exist_ok=True)

    print(f"Salvando versão experimentada na pasta: {version_str}...")

    final_output_path = os.path.join(exp_folder, "mestre.parquet")
    df_mestre.to_parquet(final_output_path, index=False)

    parquets_intermediarios = [precos_path, cvm_path, info_path, balanco_yf_path]
    for p_path in parquets_intermediarios:
        if os.path.exists(p_path):
            shutil.move(p_path, exp_folder)

    pacote_configs = {}
    pastas = ["precos", "cvm", "info", "balanco_yf"]
    for pasta in pastas:
        c_path = os.path.join(os.path.dirname(__file__), pasta, "pre_config.json")
        if os.path.exists(c_path):
            with open(c_path, "r", encoding="utf-8") as f:
                pacote_configs[pasta] = json.load(f)

    if os.path.exists(config_mestre_path):
        with open(config_mestre_path, "r", encoding="utf-8") as f:
            pacote_configs["mestre"] = json.load(f)

    exp_json_data = {
        "id_dataset": version_str,
        "aquisicao_origem": aquisicao_atual,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "notas_pesquisador": "Base gerada pelo Botão Único (Sem Data Leakage local)",
        "shape_final": [df_mestre.shape[0], df_mestre.shape[1]],
        "configs": pacote_configs,
    }

    local_json_path = os.path.join(exp_folder, f"info_{version_str}.json")
    with open(local_json_path, "w", encoding="utf-8") as f:
        json.dump(exp_json_data, f, indent=4, ensure_ascii=False)

    registry = {}
    if os.path.exists(registry_path):
        try:
            with open(registry_path, "r", encoding="utf-8") as f:
                registry = json.load(f)
        except json.JSONDecodeError:
            pass

    registry[version_str] = {
        "caminho_mestre": final_output_path,
        "timestamp": exp_json_data["timestamp"],
        "shape_final": exp_json_data["shape_final"],
    }

    with open(registry_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=4, ensure_ascii=False)

    print(f"Experimento {version_str} isolado com sucesso na pasta {exp_folder}!")
    print("Processamento finalizado com sucesso!")


def rodar_script(script_path, root_dir):
    import subprocess

    print(f"-> [INICIANDO] {script_path.name}")
    resultado = subprocess.run([sys.executable, str(script_path)], cwd=root_dir)
    if resultado.returncode != 0:
        raise RuntimeError(
            f"O script {script_path.name} falhou com código {resultado.returncode}!"
        )
    return script_path.name


if __name__ == "__main__":
    from pathlib import Path
    from concurrent.futures import ThreadPoolExecutor, as_completed

    current_dir = Path(__file__).parent.resolve()
    root_dir = current_dir.parent.parent

    scripts_fase1 = [
        root_dir / "src" / "processing" / "precos" / "process_precos.py",
        root_dir / "src" / "processing" / "cvm" / "process_cvm_dfp.py",
        root_dir / "src" / "processing" / "info" / "process_fundamentos_info.py",
        root_dir
        / "src"
        / "processing"
        / "balanco_yf"
        / "process_fundamentos_balanco.py",
    ]

    print("=== INICIANDO DATA PREP (MULTI-THREADING) ===")

    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futuros = [
                executor.submit(rodar_script, path, root_dir) for path in scripts_fase1
            ]
            for futuro in as_completed(futuros):
                script_concluido = futuro.result()
                print(f"<- [CONCLUÍDO] {script_concluido}")

        print("\n[OK] As 4 bases brutas foram processadas simultaneamente com sucesso!")
        print("\n-> [INICIANDO] O Processamento Mestre (Join)...")

        construir_tabela_mestre()
        print("\n=== PIPELINE CONCLUÍDO COM SUCESSO! ===")

    except Exception as e:
        print(f"\n[ERRO FATAL] O pipeline foi interrompido: {e}")
        sys.exit(1)
