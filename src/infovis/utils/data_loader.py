import os
import json
import sqlite3
import pandas as pd
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATALAKE_PATH = os.path.join(PROJECT_ROOT, "data", "datalake", "bateria01", "datalake.db")

@st.cache_data
def load_all_metrics():
    """
    Carrega todo o histórico de métricas do Datalake SQLite.
    Fazemos os devidos mapeamentos e JOINs para que o dataframe retornado 
    tenha o formato exato esperado pela interface gráfica retro-compatível.
    """
    if not os.path.exists(DATALAKE_PATH):
        st.error(f"Datalake não encontrado em {DATALAKE_PATH}")
        return pd.DataFrame()
        
    conn = sqlite3.connect(DATALAKE_PATH)
    
    query = """
    SELECT 
        m.id as exp_id,
        m.model_name as model_type,
        json_extract(d.generation_parameters, '$.id_dataset') as dataset_version,
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
    df = pd.read_sql(query, conn)
    conn.close()
    
    # Tratando um formato temporal_holdout da base nova para Holdout da interface velha
    # Se quiser, podemos renomear splits para ficar mais bonito na tela
    def rename_split(split):
        if not split: return "Holdout"
        if 'temporal' in split: return "Holdout"
        if 'time_series' in split: return "TSP"
        return split
        
    df['split_method'] = df['split_method'].apply(rename_split)
    
    # Remove duplicações reais, usando a chave composta correta
    if 'exp_id' in df.columns:
        df = df.drop_duplicates(subset=['exp_id'], keep='last')
        
    return df

@st.cache_data
def get_available_datasets():
    """
    Retorna os datasets únicos testados no datalake (ex: dataset001, dataset010).
    """
    df = load_all_metrics()
    if df.empty or 'dataset_version' not in df.columns:
        return []
    return sorted(df['dataset_version'].dropna().unique())

@st.cache_data
def load_xai_metadata(model_id):
    """
    Busca o array de Feature Importances injetado no SQLite para um modelo específico.
    """
    if not os.path.exists(DATALAKE_PATH):
        return {}
        
    conn = sqlite3.connect(DATALAKE_PATH)
    cur = conn.cursor()
    cur.execute("""
        SELECT feature_name, importance_value 
        FROM feature_importances 
        WHERE model_id = ?
    """, (int(model_id),))
    
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
    parquet_path = os.path.join(PROJECT_ROOT, "data", "dataset", dataset_version, "mestre.parquet")
    if not os.path.exists(parquet_path):
        return pd.DataFrame()
        
    df = pd.read_parquet(parquet_path)
    return df
