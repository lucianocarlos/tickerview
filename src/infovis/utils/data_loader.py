import os
import sqlite3
import pandas as pd
import streamlit as st

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
DATALAKE_DIR = os.path.join(PROJECT_ROOT, "data", "datalake")


@st.cache_data
def load_all_metrics():
    """
    Carrega todo o histórico de métricas de todos os bancos datalake.db
    encontrados nas subpastas de data/datalake.
    """
    if not os.path.exists(DATALAKE_DIR):
        st.error(f"Diretório Datalake não encontrado em {DATALAKE_DIR}")
        return pd.DataFrame()

    all_dfs = []

    # Varre as pastas dentro do datalake
    for folder in os.listdir(DATALAKE_DIR):
        db_path = os.path.join(DATALAKE_DIR, folder, f"{folder}.db")
        if not os.path.isfile(db_path):
            continue

        conn = sqlite3.connect(db_path)

        query = """
        SELECT 
            m.id as exp_id,
            m.model_name as model_type,
            e.target_strategy,
            json_extract(e.experiment_config, '$.split_config.method') as split_method,
            mc.val_accuracy,
            mc.val_f1_macro as val_f1_score_macro,
            mc.test_accuracy,
            mc.test_f1_macro as test_f1_score_macro,
            mc.test_f1_weighted as test_f1_score_weighted,
            mc.test_precision_macro,
            mc.test_recall_macro,
            mc.confusion_matrix,
            m.hyperparameters as parameters
        FROM models m
        JOIN experiments e ON m.experiment_id = e.id
        JOIN datasets d ON e.dataset_id = d.id
        JOIN metrics_classification mc ON m.id = mc.model_id
        """
        try:
            df = pd.read_sql(query, conn)
            df["dataset_version"] = (
                folder  # Mapeia a versão do dataset/bateria para o nome da pasta
            )
            all_dfs.append(df)
        except Exception as e:
            st.warning(f"Erro ao ler banco em {db_path}: {e}")
        finally:
            conn.close()

    if not all_dfs:
        return pd.DataFrame()

    df_global = pd.concat(all_dfs, ignore_index=True)

    # Tratando um formato temporal_holdout da base nova para Holdout da interface velha
    def rename_split(split):
        if not split:
            return "Holdout"
        if "temporal" in split:
            return "Holdout"
        if "time_series" in split:
            return "TSP"
        return split

    df_global["split_method"] = df_global["split_method"].apply(rename_split)

    # Remove duplicações reais por bateria + exp_id
    if "exp_id" in df_global.columns:
        df_global = df_global.drop_duplicates(
            subset=["dataset_version", "exp_id"], keep="last"
        )

    return df_global


@st.cache_data
def get_available_datasets():
    """
    Retorna os datasets únicos testados no datalake (ex: bateria01, bateria02).
    """
    df = load_all_metrics()
    if df.empty or "dataset_version" not in df.columns:
        return []
    return sorted(df["dataset_version"].dropna().unique())


@st.cache_data
def load_xai_metadata(model_id, dataset_version):
    """
    Busca o array de Feature Importances injetado no SQLite para um modelo e bateria específicos.
    """
    db_path = os.path.join(DATALAKE_DIR, dataset_version, f"{dataset_version}.db")
    if not os.path.exists(db_path):
        return {}

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        SELECT feature_name, importance_value 
        FROM feature_importances 
        WHERE model_id = ?
    """,
        (int(model_id),),
    )

    rows = cur.fetchall()
    conn.close()

    if rows:
        return {r[0]: r[1] for r in rows}
    return {}


@st.cache_data
def load_mestre_dataset(dataset_version="dataset001"):
    """
    Carrega o mestre.parquet do dataset requisitado para análise exploratória.
    (Como o parquet é pesado, o @st.cache_data garante que só suba para a RAM uma vez).
    """
    parquet_path = os.path.join(
        PROJECT_ROOT, "data", "dataset", dataset_version, "mestre.parquet"
    )
    if not os.path.exists(parquet_path):
        return pd.DataFrame()

    df = pd.read_parquet(parquet_path)
    return df
