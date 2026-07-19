import os
import sqlite3
import pandas as pd
import streamlit as st

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
DATALAKE_DIR = os.path.join(PROJECT_ROOT, "data", "baterias")


@st.cache_data
def load_all_metrics():
    """
    Carrega todo o histórico de métricas de todos os bancos datalake.db
    encontrados nas subpastas de data/baterias.
    """
    if not os.path.exists(DATALAKE_DIR):
        st.error(f"Diretório Datalake não encontrado em {DATALAKE_DIR}")
        return pd.DataFrame()

    all_dfs = []

    # Varre as pastas dentro de baterias
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
            e.target_strategy as target_strategy_raw,
            e.experiment_config,
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

    # Tratando target_strategy (JSON string vs Normal string)
    import json
    
    def safe_json_load(val):
        if pd.isna(val):
            return {}
        if isinstance(val, str) and val.startswith("{"):
            try:
                return json.loads(val)
            except:
                pass
        return {}

    df_global["parameters_dict"] = df_global["parameters"].apply(safe_json_load)
    df_global["experiment_config_dict"] = df_global["experiment_config"].apply(safe_json_load)

    def rename_target(target):
        if not target:
            return "unknown"
        try:
            # Se for JSON string (novo formato)
            if target.startswith("{"):
                data = json.loads(target)
                name = data.get("name", target)
                horizon = data.get("horizon_days", "")
                thresh = data.get("threshold", "")
                if horizon and thresh:
                    return f"{name} ({horizon}d | {thresh})"
                elif horizon:
                    return f"{name} ({horizon}d)"
                return name
        except:
            pass
        return target
        
    df_global["target_strategy_display"] = df_global["target_strategy"].apply(rename_target)
    df_global["target_strategy"] = df_global["target_strategy_display"]

    # Tratando model_type (Removendo subtipos)
    def rename_model(model_name):
        if not model_name:
            return "unknown"
        # Raízes conhecidas que possuem underscore
        roots = [
            "random_forest", "logistic_regression", "decision_tree", 
            "voting_classifier", "xgboost", "lightgbm", "mlp", "knn"
        ]
        for root in roots:
            if model_name.startswith(root):
                return root
        return model_name
        
    df_global["model_type"] = df_global["model_type"].apply(rename_model)

    # Remove duplicações reais por bateria + exp_id
    if "exp_id" in df_global.columns:
        df_global = df_global.drop_duplicates(
            subset=["dataset_version", "exp_id"], keep="last"
        )

    return df_global


@st.cache_data
def get_available_datasets():
    """
    Retorna os datasets únicos testados no diretório baterias (ex: bateria01, bateria02).
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
def load_bulk_xai_metadata(dataset_version):
    """
    Busca todas as importâncias de features para todos os modelos de uma bateria específica,
    retornando um DataFrame consolidado, ideal para plotagem global (Beeswarm, Matrix).
    """
    db_path = os.path.join(DATALAKE_DIR, dataset_version, f"{dataset_version}.db")
    if not os.path.exists(db_path):
        return pd.DataFrame()

    conn = sqlite3.connect(db_path)
    try:
        query = """
            SELECT f.model_id, m.model_name, f.feature_name, f.importance_value 
            FROM feature_importances f 
            JOIN models m ON f.model_id = m.id
        """
        df = pd.read_sql(query, conn)
    except Exception as e:
        st.warning(f"Erro ao carregar xai metadata bulk: {e}")
        df = pd.DataFrame()
    finally:
        conn.close()

    return df



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
