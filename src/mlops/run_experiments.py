import os
import sys

# MLOps Core Rule for Strategy A: Prevent OpenMP / C++ Thread Oversubscription
# Isso DEVE ser configurado antes de qualquer import do numpy/torch/sklearn/xgboost
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import time
import pandas as pd
import platform
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed
from collections import defaultdict
import warnings
import threading
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

# Importações locais do MLOps
import db_manager
from classification import classifier

# Suprimir o aviso de parada de worker do loky (Comum ao usar PyTorch via Multiprocessing)
warnings.filterwarnings("ignore", category=UserWarning, module="joblib.externals.loky")


class TaskDispatcher:
    def __init__(self):
        self.gpu_exclusive = []
        self.gpu_native = []
        self.cpu_native = []
        self.cpu_exclusive = []
        self.lock = threading.Lock()

    def add_task(self, pref_queue, cache_path, chunk_modelos, global_permutation):
        with self.lock:
            task = (cache_path, chunk_modelos, global_permutation)
            if pref_queue == "gpu_exclusive":
                self.gpu_exclusive.append(task)
            elif pref_queue == "gpu":
                self.gpu_native.append(task)
            elif pref_queue == "cpu_exclusive":
                self.cpu_exclusive.append(task)
            else:  # defaults to "cpu"
                self.cpu_native.append(task)

    def get_task_for_gpu(self):
        with self.lock:
            if self.gpu_exclusive:
                return ("GPU-EXCLUSIVE", self.gpu_exclusive.pop(0))
            if self.gpu_native:
                return ("GPU-NATIVE", self.gpu_native.pop(0))
            if self.cpu_native:
                return ("GPU-STEAL", self.cpu_native.pop(0))
            return None

    def get_task_for_cpu(self):
        with self.lock:
            if self.cpu_exclusive:
                return ("CPU-EXCLUSIVE", self.cpu_exclusive.pop(0))
            if self.cpu_native:
                return ("CPU-NATIVE", self.cpu_native.pop(0))
            if self.gpu_native:
                return ("CPU-STEAL", self.gpu_native.pop(0))
            return None


def worker_preprocessamento(
    raw_data_path,
    strategy,
    target_def,
    split_config,
    inf_strat,
    imp_strategy,
    out_method,
    scale_method,
    cache_path,
    exp_id,
):
    import joblib
    import os
    import pandas as pd

    if os.path.exists(cache_path):
        return {"status": "skipped", "exp_id": exp_id}

    try:
        df_raw = pd.read_parquet(raw_data_path)
        X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = (
            classifier.preprocessar_dados(
                df_raw,
                strategy,
                target_def,
                split_config,
                inf_strat,
                imp_strategy,
                out_method,
                scale_method,
            )
        )
        joblib.dump(
            (X_train, y_train, X_val, y_val, X_test, y_test, feature_cols), cache_path
        )
        return {"status": "success", "exp_id": exp_id}
    except Exception as e:
        return {"status": "error", "exp_id": exp_id, "error_msg": str(e)}


def worker_treinar_lote(cache_path, modelos_lista, calc_permutation):
    """Treina todos os modelos de um mesmo experimento com uma única leitura de disco."""
    import joblib

    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = joblib.load(
        cache_path
    )

    # Cache de Feature Selection: evita re-treinar a árvore para cada modelo
    fs_cache = {}
    resultados = []

    for model_name, params, exp_id in modelos_lista:
        model_start = time.time()
        try:
            metrics, conf_mat, feature_importances, importance_type = (
                classifier.treinar_e_avaliar_modelo_pre_processado(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    X_test=X_test,
                    y_test=y_test,
                    feature_cols=feature_cols,
                    model_name=model_name,
                    hparams=params,
                    calculate_permutation_importance=calc_permutation,
                    fs_cache=fs_cache,
                )
            )
            model_time = time.time() - model_start
            resultados.append(
                {
                    "status": "success",
                    "exp_id": exp_id,
                    "model_name": model_name,
                    "params": params,
                    "metrics": metrics,
                    "conf_mat": conf_mat,
                    "feature_importances": feature_importances,
                    "importance_type": importance_type,
                    "model_time": model_time,
                }
            )
        except Exception as e:
            resultados.append(
                {
                    "status": "error",
                    "exp_id": exp_id,
                    "model_name": model_name,
                    "params": params,
                    "error_msg": str(e),
                }
            )
    return resultados


def carregar_configuracao(config_path):
    """Carrega arquivo de configuração suportando formatos YAML (.yaml/.yml) e JSON (.json)."""
    ext = os.path.splitext(config_path)[1].lower()
    with open(config_path, "r", encoding="utf-8") as f:
        if ext in [".yaml", ".yml"]:
            if yaml is None:
                raise ImportError("PyYAML não instalado. Execute 'pip install pyyaml' para ler arquivos .yaml")
            return yaml.safe_load(f)
        else:
            return json.load(f)


def executar_bateria_teste(config_path):
    battery_start_time = time.time()

    config = carregar_configuracao(config_path)

    # NOVO: Chave mestra de performance
    global_permutation = config.get("calculate_permutation_importance", False)

    # O nome da bateria vem explicitamente de dentro do arquivo (ou usa o nome do arquivo sem extensão)
    file_name_fallback = os.path.splitext(os.path.basename(config_path))[0]
    battery_name = config.get(
        "baterias_name", config.get("bateria_id", file_name_fallback)
    )

    hardware_config = config.get("hardware_config", {})
    cpu_workers_limit = hardware_config.get("cpu_workers", 13)
    gpu_workers_limit = hardware_config.get("gpu_workers", 3)
    chunk_size_limit = hardware_config.get("chunk_size", 15)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    dynamic_db_path = os.path.join(
        project_root, "data", "baterias", battery_name, f"{battery_name}.db"
    )
    db_manager.set_db_path(dynamic_db_path)

    # Verifica se já existe um checkpoint Pai (a própria Bateria)
    battery_id = db_manager.get_or_create_battery(battery_name, config)
    print("=" * 50)
    print("      MLOps 2.0 - SQLite Relacional Datalake      ")
    print("=" * 50)
    print(
        f"[BATTERY] Bateria Global Iniciada/Retomada! ID: {battery_id} | Name: {battery_name}\n"
    )

    dataset_vs = config.get("datasets", ["dataset003"])

    for dataset_v in dataset_vs:
        print(f"\n[INFO] Iniciando Pipeline para o Dataset: {dataset_v}")

        # 1. Carregar Dados Brutos e Info do Dataset
        dataset_path = os.path.join(
            project_root, "data", "dataset", dataset_v, "mestre.parquet"
        )
        info_dataset_path = os.path.join(
            project_root, "data", "dataset", dataset_v, f"info_{dataset_v}.json"
        )

        if not os.path.exists(dataset_path):
            print(f"[ERRO] {dataset_path} não encontrado.")
            continue

        df_raw = pd.read_parquet(dataset_path)
        df_raw["Date"] = pd.to_datetime(df_raw["Date"])

        gen_params = {}
        if os.path.exists(info_dataset_path):
            with open(info_dataset_path, "r", encoding="utf-8") as f:
                gen_params = json.load(f)

        # 2. Registrar Dataset no SQLite
        dataset_id = db_manager.get_or_create_dataset(
            version_hash=dataset_v,
            features_count=len(df_raw.columns),
            rows_count=len(df_raw),
            generation_parameters=gen_params,
        )
        print(f"       -> Dataset cadastrado/recuperado no Banco! ID: {dataset_id}")

        # Salvar dados brutos uma vez no disco para evitar serialização massiva via Joblib
        cache_dir = os.path.join(
            project_root, "data", "baterias", battery_name, "cache"
        )
        os.makedirs(cache_dir, exist_ok=True)
        raw_cache_path = os.path.join(cache_dir, f"raw_{dataset_v}.parquet")
        df_raw.to_parquet(raw_cache_path)

        split_strategies = config.get(
            "split_configs", [{"method": "temporal_holdout", "train_ratio": 0.7}]
        )
        target_strategies = config.get("target_strategies", ["outperformance"])

        prep = config.get("preprocessing", {})
        inf_strategies = prep.get("inf_handling", ["replace_nan"])
        imputation_strategies = prep.get("imputation", ["fill_zero"])
        outlier_methods = prep.get("outlier_handling", ["none"])
        scaling_methods = prep.get("scaling", ["none"])
        models_config = config.get("models", {})
        target_def = config.get(
            "target_definition", {"horizon_days": 10, "threshold": 0.02}
        )

        prep_tasks = []
        todas_as_tasks_agrupadas = defaultdict(
            lambda: defaultdict(list)
        )  # cache_path -> pref_queue -> list of models
        exp_ids_to_update = set()

        for strategy_item in target_strategies:
            if isinstance(strategy_item, dict):
                strategy_name = strategy_item.get("name")
                current_target_def = {
                    k: v for k, v in strategy_item.items() if k != "name"
                }
                db_target_strategy = strategy_item
            else:
                strategy_name = strategy_item
                current_target_def = target_def
                db_target_strategy = strategy_item

            for split_config in split_strategies:
                for inf_strat in inf_strategies:
                    for imp_strategy in imputation_strategies:
                        for out_method in outlier_methods:
                            for scale_method in scaling_methods:
                                print(
                                    f"\n[EXPERIMENT] Target: {db_target_strategy} | Inf: {inf_strat} | Imp: {imp_strategy} | Out: {out_method} | Scale: {scale_method}"
                                )

                                experiment_config = {
                                    "target_definition": current_target_def,
                                    "split_config": split_config,
                                    "inf_handling_strategy": inf_strat,
                                    "imputation_strategy": imp_strategy,
                                    "outlier_handling": out_method,
                                    "scaling_method": scale_method,
                                    "grid_search_config": models_config,
                                }

                                exp_id = db_manager.get_or_create_experiment(
                                    battery_id=battery_id,
                                    dataset_id=dataset_id,
                                    task_type="classification",
                                    target_strategy=db_target_strategy,
                                    experiment_config=experiment_config,
                                )
                                exp_ids_to_update.add(exp_id)

                                cache_path = os.path.join(
                                    cache_dir, f"exp_{exp_id}.joblib"
                                )

                                prep_tasks.append(
                                    (
                                        raw_cache_path,
                                        strategy_name,
                                        current_target_def,
                                        split_config,
                                        inf_strat,
                                        imp_strategy,
                                        out_method,
                                        scale_method,
                                        cache_path,
                                        exp_id,
                                    )
                                )
                                trained_models_set = db_manager.get_trained_models(
                                    exp_id
                                )

                                for model_name, param_grid_raw in models_config.items():
                                    if model_name.startswith("_"):
                                        continue

                                    param_grid = param_grid_raw.copy()
                                    pref_queue = param_grid.pop("pref_queue", "cpu")

                                    clean_param_grid = {
                                        k: v
                                        for k, v in param_grid.items()
                                        if not k.startswith("_")
                                    }
                                    grid = list(ParameterGrid(clean_param_grid))

                                    for params in grid:
                                        hparams_str = (
                                            json.dumps(
                                                params,
                                                ensure_ascii=False,
                                                sort_keys=True,
                                            )
                                            if isinstance(params, dict)
                                            else params
                                        )
                                        if (
                                            model_name,
                                            hparams_str,
                                        ) not in trained_models_set:
                                            task_tuple = (model_name, params, exp_id)
                                            todas_as_tasks_agrupadas[cache_path][
                                                pref_queue
                                            ].append(task_tuple)
                                        else:
                                            print(
                                                f"   -> [SKIPPED] {model_name} já treinado nesta arquitetura."
                                            )

        if prep_tasks:
            print(
                f"\n[INFO] ESTÁGIO 1: Pré-processamento Paralelo ({len(prep_tasks)} matrizes)..."
            )
            prep_start = time.time()
            resultados_prep = Parallel(n_jobs=-1, return_as="generator_unordered")(
                delayed(worker_preprocessamento)(*task) for task in prep_tasks
            )
            for res in resultados_prep:
                if res["status"] == "error":
                    print(
                        f"   -> [ERRO] Falha no pré-processamento do exp {res['exp_id']}: {res['error_msg']}"
                    )
            print(
                f"   -> [ESTÁGIO 1 CONCLUÍDO] Todos os Caches criados em {time.time() - prep_start:.1f}s."
            )

        # Count total tasks
        total_tarefas = 0
        for cp, prefs in todas_as_tasks_agrupadas.items():
            for pref_q, modelos in prefs.items():
                total_tarefas += len(modelos)

        if total_tarefas == 0:
            print(
                "\n[INFO] Todos os modelos já foram treinados neste dataset (100% Retomado). Pulando..."
            )
            continue

        print(
            f"\n[INFO] ESTÁGIO 2: Treinamento Paralelo ({total_tarefas} tarefas). TaskDispatcher iniciado às {time.time()}"
        )

        parallel_start = time.time()

        # Configura o Dispatcher com lotes menores para evitar Tail-End Starvation
        dispatcher = TaskDispatcher()
        chunk_size = chunk_size_limit
        for cp, prefs in todas_as_tasks_agrupadas.items():
            for pref_q, modelos in prefs.items():
                for i in range(0, len(modelos), chunk_size):
                    chunk = modelos[i : i + chunk_size]
                    dispatcher.add_task(pref_q, cp, chunk, global_permutation)

        db_lock = threading.Lock()
        modelos_treinados = 0

        def run_worker_pool(pool_name, max_workers, get_task_fn, initializer=None, initargs=()):
            print(f"\n[INFO] INICIANDO Pool {pool_name} ({max_workers} workers)...")
            with ProcessPoolExecutor(max_workers=max_workers, initializer=initializer, initargs=initargs) as executor:
                futures_map = {}

                while True:
                    # Tenta manter o pool sempre cheio
                    while len(futures_map) < max_workers:
                        task_obj = get_task_fn()
                        if task_obj is None:
                            break

                        queue_type, (cp, chunk, g_perm) = task_obj
                        future = executor.submit(worker_treinar_lote, cp, chunk, g_perm)
                        futures_map[future] = queue_type

                    if not futures_map:
                        break  # Acabaram as tarefas no Dispatcher

                    # Espera a PRIMEIRA tarefa finalizar
                    done, not_done = wait(
                        futures_map.keys(),
                        return_when=FIRST_COMPLETED,
                    )

                    for future in done:
                        q_type = futures_map.pop(future)
                        try:
                            lote_res = future.result()
                            for res in lote_res:
                                exp_id_res = res["exp_id"]
                                if res["status"] == "success":
                                    with db_lock:
                                        db_manager.save_model_results(
                                            experiment_id=exp_id_res,
                                            model_name=res["model_name"],
                                            hyperparameters=res["params"],
                                            exec_time_sec=res["model_time"],
                                            metrics_class=res["metrics"],
                                            confusion_matrix=res["conf_mat"],
                                            feature_importances=res[
                                                "feature_importances"
                                            ],
                                            importance_type=res["importance_type"],
                                            hardware_used=pool_name,
                                        )
                                    with db_lock:
                                        nonlocal modelos_treinados
                                        modelos_treinados += 1

                                    elapsed = time.time() - parallel_start
                                    vel = modelos_treinados / elapsed
                                    rem = total_tarefas - modelos_treinados
                                    eta = rem / vel if vel > 0 else 0
                                    eta_m, eta_s = divmod(int(eta), 60)

                                    print(
                                        f"   -> [{pool_name[:3].upper()}:{q_type}] Exp {exp_id_res}: {res['model_name']} salvo. F1: {res['metrics']['test_f1_macro']:.4f} | Tempo: {res['model_time']:.1f}s | Modelos: {modelos_treinados}/{total_tarefas} | ETA: {eta_m}m {eta_s}s"
                                    )
                                elif res["status"] == "error":
                                    print(
                                        f"   -> [ERRO FATAL] Exp {exp_id_res}: Falha no {res['model_name']}: {res['error_msg']}"
                                    )
                        except Exception as e:
                            print(
                                f"[ERRO LOTE] Erro ao processar resultado do pool: {e}"
                            )

            print(
                f"[INFO] Pool {pool_name} finalizado (Fila completamente esgotada para ele)."
            )

        def init_gpu_env(d_id):
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = str(d_id)

        def run_gpu(device_id, workers):
            gpu_name = f"gpu-{device_id}"
            try:
                import torch
                if torch.cuda.is_available() and device_id < torch.cuda.device_count():
                    gpu_name = f"gpu-{torch.cuda.get_device_name(device_id).lower()}"
            except Exception:
                pass
            run_worker_pool(gpu_name, workers, dispatcher.get_task_for_gpu, initializer=init_gpu_env, initargs=(device_id,))

        def run_cpu():
            cpu_name = "cpu"
            try:
                import platform
                proc_info = platform.processor() or "cpu"
                cpu_name = f"cpu-{proc_info.lower()}"
            except Exception:
                pass
            
            def init_cpu_env():
                import os
                # Impede que a CPU enxergue qualquer GPU acidentalmente
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                
            run_worker_pool(cpu_name, cpu_workers_limit, dispatcher.get_task_for_cpu, initializer=init_cpu_env)

        threads = []
        
        # Inicia pools de GPU(s)
        if isinstance(gpu_workers_limit, list):
            for i, w in enumerate(gpu_workers_limit):
                if w > 0:
                    t = threading.Thread(target=run_gpu, args=(i, w))
                    threads.append(t)
                    t.start()
        else:
            if gpu_workers_limit > 0:
                t = threading.Thread(target=run_gpu, args=(0, gpu_workers_limit))
                threads.append(t)
                t.start()

        # Inicia pool de CPU
        if cpu_workers_limit > 0:
            t_cpu = threading.Thread(target=run_cpu)
            threads.append(t_cpu)
            t_cpu.start()

        # Aguarda todas as threads
        for t in threads:
            t.join()

        parallel_time = time.time() - parallel_start
        for eid in exp_ids_to_update:
            db_manager.update_experiment_time(eid, parallel_time)

        print(
            f"[OK] {modelos_treinados} modelos computados assincronamente em {parallel_time:.1f}s."
        )

    # FECHAMENTO DA BATERIA GLOBAL
    battery_total_time = time.time() - battery_start_time
    db_manager.finish_battery(battery_id, battery_total_time)
    print(
        f"\n[SUCESSO ABSOLUTO] Bateria Mãe (ID: {battery_id}) encerrada com sucesso total em {battery_total_time:.1f} segundos!"
    )

    # ---------------------------------------------------------
    # SUMÁRIO FINAL DA BATERIA
    # ---------------------------------------------------------
    try:
        from datetime import timedelta

        conn = db_manager.get_connection()

        # Datetimes
        cur = conn.cursor()
        cur.execute(
            "SELECT created_at, finished_at FROM batteries WHERE id = ?", (battery_id,)
        )
        row = cur.fetchone()
        dt_start = row[0] if row else "?"
        dt_end = row[1] if row else "?"

        time_str = str(timedelta(seconds=int(battery_total_time)))

        # Número de modelos
        cur.execute(
            "SELECT COUNT(id) FROM models WHERE experiment_id IN (SELECT id FROM experiments WHERE battery_id = ?)",
            (battery_id,),
        )
        num_models = cur.fetchone()[0]

        # Max F1 per target
        df_tgt = pd.read_sql(
            f"""
            SELECT e.target_strategy, MAX(mc.test_f1_macro) as max_f1
            FROM models m
            JOIN metrics_classification mc ON m.id = mc.model_id
            JOIN experiments e ON m.experiment_id = e.id
            WHERE e.battery_id = {battery_id}
            GROUP BY e.target_strategy
        """,
            conn,
        )

        # Max F1 per estimator
        df_est = pd.read_sql(
            f"""
            SELECT m.model_name, MAX(mc.test_f1_macro) as max_f1
            FROM models m
            JOIN metrics_classification mc ON m.id = mc.model_id
            JOIN experiments e ON m.experiment_id = e.id
            WHERE e.battery_id = {battery_id}
            GROUP BY m.model_name
        """,
            conn,
        )

        print("\n" + "=" * 60)
        print("               RELATÓRIO FINAL DA BATERIA")
        print("=" * 60)
        print(f"Data/Hora Início:     {dt_start}")
        print(f"Data/Hora Término:    {dt_end}")
        print(f"Tempo Total Gasto:    {time_str}")
        print(f"Total de Modelos:     {num_models}")
        print("\n--- F1 Score Máximo (Por Target) ---")
        for _, r in df_tgt.iterrows():
            tgt_name = (
                json.loads(r["target_strategy"]).get("name")
                if "{" in r["target_strategy"]
                else r["target_strategy"]
            )
            print(f" {tgt_name:<30} | {r['max_f1']:.4f}")

        print("\n--- F1 Score Máximo (Por Estimador) ---")
        for _, r in df_est.iterrows():
            print(f" {r['model_name']:<30} | {r['max_f1']:.4f}")

        print("=" * 60 + "\n")

    except Exception as e:
        print(f"[AVISO] Não foi possível gerar o relatório final: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        caminho_config = sys.argv[1]
    else:
        # Carregamento autônomo: prefere .yaml se existir, senão usa .json
        caminho_yaml = os.path.join(os.path.dirname(__file__), "conf_exp_opt.yaml")
        caminho_json = os.path.join(os.path.dirname(__file__), "conf_exp_opt.json")
        if os.path.exists(caminho_yaml):
            caminho_config = caminho_yaml
        else:
            caminho_config = caminho_json
        print(f"[SETUP] Nenhum parâmetro fornecido. Carregando padrão: {caminho_config}")

    executar_bateria_teste(caminho_config)
