import sqlite3
import json
import os

DB_PATH = r"data\datalake\default\datalake.db"

def set_db_path(new_path):
    """Permite que o Orquestrador defina o destino do Datalake dinamicamente"""
    global DB_PATH
    DB_PATH = new_path

def _init_db(conn):
    cur = conn.cursor()
    cur.execute('''CREATE TABLE IF NOT EXISTS datasets (id INTEGER PRIMARY KEY AUTOINCREMENT, version_hash TEXT NOT NULL UNIQUE, features_count INTEGER, rows_count INTEGER, generation_parameters TEXT, created_at DATETIME DEFAULT (datetime('now', 'localtime')))''')
    cur.execute('''CREATE TABLE IF NOT EXISTS batteries (id INTEGER PRIMARY KEY AUTOINCREMENT, battery_name TEXT, global_config TEXT, elapsed_time_sec FLOAT, created_at DATETIME DEFAULT (datetime('now', 'localtime')), finished_at DATETIME)''')
    cur.execute('''CREATE TABLE IF NOT EXISTS experiments (id INTEGER PRIMARY KEY AUTOINCREMENT, battery_id INTEGER NOT NULL, dataset_id INTEGER NOT NULL, task_type TEXT NOT NULL, target_strategy TEXT, experiment_config TEXT, elapsed_time_sec FLOAT, created_at DATETIME DEFAULT (datetime('now', 'localtime')), FOREIGN KEY (battery_id) REFERENCES batteries (id), FOREIGN KEY (dataset_id) REFERENCES datasets (id))''')
    cur.execute('''CREATE TABLE IF NOT EXISTS models (id INTEGER PRIMARY KEY AUTOINCREMENT, experiment_id INTEGER NOT NULL, model_name TEXT NOT NULL, hyperparameters TEXT, execution_time_sec FLOAT, created_at DATETIME DEFAULT (datetime('now', 'localtime')), FOREIGN KEY (experiment_id) REFERENCES experiments (id))''')
    cur.execute('''CREATE TABLE IF NOT EXISTS metrics_classification (model_id INTEGER PRIMARY KEY, val_accuracy FLOAT, val_f1_macro FLOAT, test_accuracy FLOAT, test_f1_macro FLOAT, test_f1_weighted FLOAT, test_precision_macro FLOAT, test_recall_macro FLOAT, confusion_matrix TEXT, FOREIGN KEY (model_id) REFERENCES models (id))''')
    cur.execute('''CREATE TABLE IF NOT EXISTS metrics_clustering (model_id INTEGER PRIMARY KEY, silhouette_score FLOAT, davies_bouldin FLOAT, FOREIGN KEY (model_id) REFERENCES models (id))''')
    cur.execute('''CREATE TABLE IF NOT EXISTS metrics_association (model_id INTEGER PRIMARY KEY, support_avg FLOAT, confidence_avg FLOAT, FOREIGN KEY (model_id) REFERENCES models (id))''')
    cur.execute('''CREATE TABLE IF NOT EXISTS feature_importances (id INTEGER PRIMARY KEY AUTOINCREMENT, model_id INTEGER NOT NULL, feature_name TEXT NOT NULL, importance_value FLOAT NOT NULL, importance_type TEXT NOT NULL, FOREIGN KEY (model_id) REFERENCES models (id))''')
    conn.commit()

_DB_INITIALIZED = False

def get_connection():
    global _DB_INITIALIZED
    # Cria os diretórios caso não existam
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    if not _DB_INITIALIZED:
        _init_db(conn)
        _DB_INITIALIZED = True
    return conn

def get_or_create_dataset(version_hash, features_count, rows_count, generation_parameters):
    """Retorna o ID do dataset. Se não existir, insere no banco."""
    conn = get_connection()
    cur = conn.cursor()
    
    cur.execute("SELECT id FROM datasets WHERE version_hash = ?", (version_hash,))
    row = cur.fetchone()
    
    if row:
        dataset_id = row[0]
    else:
        gen_params_str = json.dumps(generation_parameters, ensure_ascii=False) if isinstance(generation_parameters, dict) else generation_parameters
        cur.execute('''
            INSERT INTO datasets (version_hash, features_count, rows_count, generation_parameters)
            VALUES (?, ?, ?, ?)
        ''', (version_hash, features_count, rows_count, gen_params_str))
        dataset_id = cur.lastrowid
        conn.commit()
    
    conn.close()
    return dataset_id

def get_or_create_battery(battery_name, global_config):
    """Retorna a bateria existente pelo nome ou cria uma nova (Checkpoint Pai)."""
    conn = get_connection()
    cur = conn.cursor()
    
    cur.execute("SELECT id FROM batteries WHERE battery_name = ?", (battery_name,))
    row = cur.fetchone()
    if row:
        bat_id = row[0]
    else:
        config_str = json.dumps(global_config, ensure_ascii=False) if isinstance(global_config, dict) else global_config
        cur.execute('''
            INSERT INTO batteries (battery_name, global_config)
            VALUES (?, ?)
        ''', (battery_name, config_str))
        bat_id = cur.lastrowid
        conn.commit()
        
    conn.close()
    return bat_id

def update_battery_time(battery_id, elapsed_time_sec):
    """Atualiza o tempo total da bateria global."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE batteries SET elapsed_time_sec = ? WHERE id = ?", (elapsed_time_sec, battery_id))
    conn.commit()
    conn.close()

def finish_battery(battery_id, elapsed_time_sec):
    """Atualiza o tempo total e marca o término da bateria (finished_at) com a hora local."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute('''
        UPDATE batteries 
        SET elapsed_time_sec = ?, 
            finished_at = datetime('now', 'localtime') 
        WHERE id = ?
    ''', (elapsed_time_sec, battery_id))
    conn.commit()
    conn.close()

def get_or_create_experiment(battery_id, dataset_id, task_type, target_strategy, experiment_config):
    """Retorna o experimento se já existir com a mesma configuração exata, ou cria novo (Checkpoint Filho)."""
    conn = get_connection()
    cur = conn.cursor()
    
    config_str = json.dumps(experiment_config, ensure_ascii=False) if isinstance(experiment_config, dict) else experiment_config
    target_strategy_str = json.dumps(target_strategy, ensure_ascii=False) if isinstance(target_strategy, dict) else target_strategy
    
    cur.execute('''
        SELECT id FROM experiments 
        WHERE battery_id = ? AND dataset_id = ? AND task_type = ? AND target_strategy = ? AND experiment_config = ?
    ''', (battery_id, dataset_id, task_type, target_strategy_str, config_str))
    
    row = cur.fetchone()
    if row:
        exp_id = row[0]
    else:
        cur.execute('''
            INSERT INTO experiments (battery_id, dataset_id, task_type, target_strategy, experiment_config)
            VALUES (?, ?, ?, ?, ?)
        ''', (battery_id, dataset_id, task_type, target_strategy_str, config_str))
        exp_id = cur.lastrowid
        conn.commit()
        
    conn.close()
    return exp_id

def update_experiment_time(experiment_id, elapsed_time_sec):
    """Atualiza o tempo total gasto na bateria."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("UPDATE experiments SET elapsed_time_sec = ? WHERE id = ?", (elapsed_time_sec, experiment_id))
    conn.commit()
    conn.close()

def model_exists(experiment_id, model_name, hyperparameters):
    """Verifica se esse exato modelo já foi computado nesta matriz arquitetônica (Checkpoint de Modelo)."""
    conn = get_connection()
    cur = conn.cursor()
    hparams_str = json.dumps(hyperparameters, ensure_ascii=False) if isinstance(hyperparameters, dict) else hyperparameters
    
    cur.execute('''
        SELECT id FROM models
        WHERE experiment_id = ? AND model_name = ? AND hyperparameters = ?
    ''', (experiment_id, model_name, hparams_str))
    
    row = cur.fetchone()
    conn.close()
    return row is not None

def get_trained_models(experiment_id):
    """Retorna um set com (model_name, hyperparameters_str) dos modelos já treinados para um experimento."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute('''
        SELECT model_name, hyperparameters FROM models
        WHERE experiment_id = ?
    ''', (experiment_id,))
    
    rows = cur.fetchall()
    conn.close()
    
    # Retorna como um set de tuplas para checagem O(1) em memória
    return set((row[0], row[1]) for row in rows)

def save_model_results(experiment_id, model_name, hyperparameters, exec_time_sec, 
                       metrics_class, confusion_matrix, feature_importances, importance_type="entropy"):
    """
    Salva o modelo, as métricas e o XAI no banco em uma única transação segura.
    """
    conn = get_connection()
    cur = conn.cursor()
    try:
        # 1. Inserir o Modelo
        hparams_str = json.dumps(hyperparameters, ensure_ascii=False) if isinstance(hyperparameters, dict) else hyperparameters
        cur.execute('''
            INSERT INTO models (experiment_id, model_name, hyperparameters, execution_time_sec)
            VALUES (?, ?, ?, ?)
        ''', (experiment_id, model_name, hparams_str, exec_time_sec))
        
        model_id = cur.lastrowid
        
        # 2. Inserir as Métricas de Classificação
        cm_str = json.dumps(confusion_matrix) if isinstance(confusion_matrix, list) else confusion_matrix
        cur.execute('''
            INSERT INTO metrics_classification 
            (model_id, val_accuracy, val_f1_macro, test_accuracy, test_f1_macro, test_f1_weighted, test_precision_macro, test_recall_macro, confusion_matrix)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            model_id, 
            metrics_class.get('val_accuracy'), 
            metrics_class.get('val_f1_macro'),
            metrics_class.get('test_accuracy'),
            metrics_class.get('test_f1_macro'),
            metrics_class.get('test_f1_weighted'),
            metrics_class.get('test_precision_macro'),
            metrics_class.get('test_recall_macro'),
            cm_str
        ))
        
        # 3. Inserir Feature Importances (O XAI Mestre)
        if feature_importances:
            xai_records = [
                (model_id, feat_name, float(val), importance_type) 
                for feat_name, val in feature_importances.items()
            ]
            cur.executemany('''
                INSERT INTO feature_importances (model_id, feature_name, importance_value, importance_type)
                VALUES (?, ?, ?, ?)
            ''', xai_records)
            
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()
