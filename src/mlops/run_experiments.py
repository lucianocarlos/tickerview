import os
import sys
import json
import time
import pandas as pd
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed
from collections import defaultdict
import warnings
import threading
import queue
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import joblib

# Importações locais do MLOps
import db_manager
from classification import classifier

# Suprimir o aviso de parada de worker do loky (Comum ao usar PyTorch via Multiprocessing)
warnings.filterwarnings("ignore", category=UserWarning, module="joblib.externals.loky")


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


def worker_single_model(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    feature_cols,
    model_name,
    params,
    exp_id,
    calc_permutation,
    fs_cache,
    force_cpu=False,
):
    """Treina um único modelo em uma thread separada. Utilizado no Steal Phase da CPU."""
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
                force_cpu=force_cpu,
            )
        )
        model_time = time.time() - model_start
        return [
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
        ]
    except Exception as e:
        return [
            {
                "status": "error",
                "exp_id": exp_id,
                "model_name": model_name,
                "error_msg": str(e),
            }
        ]


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


def executar_bateria_teste(config_path):
    battery_start_time = time.time()

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # NOVO: Chave mestra de performance
    global_permutation = config.get("calculate_permutation_importance", False)

    # O nome do Datalake vem explicitamente de dentro do JSON (ou usa o nome do arquivo se não existir)
    file_name_fallback = os.path.basename(config_path).replace(".json", "")
    battery_name = config.get("datalake_name", file_name_fallback)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    dynamic_db_path = os.path.join(project_root, "data", "datalake", battery_name, "datalake.db")
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
        cache_dir = os.path.join(project_root, "data", "datalake", battery_name, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        raw_cache_path = os.path.join(cache_dir, f"raw_{dataset_v}.parquet")
        df_raw.to_parquet(raw_cache_path)

        split_strategies = config.get(
            "split_strategies", [{"method": "temporal_holdout", "train_ratio": 0.7}]
        )
        target_strategies = config.get("target_strategies", ["outperformance"])
        inf_strategies = config.get("inf_handling_strategies", ["replace_nan"])
        imputation_strategies = config.get("imputation_strategies", ["fill_zero"])
        outlier_methods = config.get("outlier_handling_methods", ["none"])
        scaling_methods = config.get("scaling_methods", ["none"])
        models_config = config.get("models", {})
        target_def = config.get(
            "target_definition", {"horizon_days": 10, "threshold": 0.02}
        )

        prep_tasks = []
        todas_as_tasks = []
        tarefas_cpu = []
        tarefas_gpu = []
        exp_ids_to_update = set()

        for strategy in target_strategies:
            for split_config in split_strategies:
                for inf_strat in inf_strategies:
                    for imp_strategy in imputation_strategies:
                        for out_method in outlier_methods:
                            for scale_method in scaling_methods:
                                print(
                                    f"\n[EXPERIMENT] Target: {strategy} | Inf: {inf_strat} | Imp: {imp_strategy} | Out: {out_method} | Scale: {scale_method}"
                                )

                                experiment_config = {
                                    "target_definition": target_def,
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
                                    target_strategy=strategy,
                                    experiment_config=experiment_config,
                                )
                                exp_ids_to_update.add(exp_id)

                                cache_path = os.path.join(
                                    cache_dir, f"exp_{exp_id}.joblib"
                                )

                                prep_tasks.append(
                                    (
                                        raw_cache_path,
                                        strategy,
                                        target_def,
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

                                for model_name, param_grid in models_config.items():
                                    if model_name.startswith("_"):
                                        continue

                                    clean_param_grid = {
                                        k: v
                                        for k, v in param_grid.items()
                                        if not k.startswith("_")
                                    }
                                    grid = list(ParameterGrid(clean_param_grid))

                                    for params in grid:
                                        hparams_str = (
                                            json.dumps(params, ensure_ascii=False)
                                            if isinstance(params, dict)
                                            else params
                                        )
                                        if (
                                            model_name,
                                            hparams_str,
                                        ) not in trained_models_set:
                                            task_tuple = (
                                                cache_path,
                                                model_name,
                                                params,
                                                exp_id,
                                                global_permutation,
                                            )
                                            todas_as_tasks.append(task_tuple)
                                            if model_name in [
                                                "knn",
                                                "mlp",
                                                "logistic_regression",
                                                "voting_classifier",
                                            ]:
                                                tarefas_gpu.append(task_tuple)
                                            else:
                                                tarefas_cpu.append(task_tuple)
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

        if not todas_as_tasks:
            print(
                "\n[INFO] Todos os modelos já foram treinados neste dataset (100% Retomado). Pulando..."
            )
            continue

        print(
            f"\n[INFO] ESTÁGIO 2: Treinamento Paralelo. Disparando {len(todas_as_tasks)} tarefas para a RTX 2050..."
        )
        parallel_start = time.time()

        # Preparar lotes globais
        lotes_cpu_dict = defaultdict(list)
        for cache_path, model_name, params, exp_id, _ in tarefas_cpu:
            lotes_cpu_dict[cache_path].append((model_name, params, exp_id))

        lotes_gpu_dict = defaultdict(list)
        for cache_path, model_name, params, exp_id, _ in tarefas_gpu:
            lotes_gpu_dict[cache_path].append((model_name, params, exp_id))

        lotes_totais = len(lotes_cpu_dict) + len(lotes_gpu_dict)
        total_tarefas = len(tarefas_cpu) + len(tarefas_gpu)
        print(
            f"       -> {len(lotes_cpu_dict)} lotes nativos CPU, {len(lotes_gpu_dict)} lotes nativos GPU"
        )

        queue_gpu = queue.Queue()
        for cp, modelos in lotes_gpu_dict.items():
            queue_gpu.put((cp, modelos))

        db_lock = threading.Lock()
        modelos_treinados = 0
        lotes_concluidos = 0

        def processar_resultados_futures(futures, stage_name, num_lotes):
            nonlocal modelos_treinados, lotes_concluidos
            lotes_do_estagio = 0
            for future in as_completed(futures):
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
                                feature_importances=res["feature_importances"],
                                importance_type=res["importance_type"],
                            )
                        modelos_treinados += 1

                        elapsed = time.time() - parallel_start
                        vel = modelos_treinados / elapsed
                        rem = total_tarefas - modelos_treinados
                        eta = rem / vel if vel > 0 else 0
                        eta_m, eta_s = divmod(int(eta), 60)

                        print(
                            f"   -> [{stage_name}] Exp {exp_id_res}: {res['model_name']} salvo. F1: {res['metrics']['test_f1_macro']:.4f} | Tempo: {res['model_time']:.1f}s | ETA: {eta_m}m {eta_s}s"
                        )
                    elif res["status"] == "error":
                        print(
                            f"   -> [ERRO FATAL] Exp {exp_id_res}: Falha no {res['model_name']}: {res['error_msg']}"
                        )

                with db_lock:
                    lotes_concluidos += 1
                    lotes_do_estagio += 1

                if lotes_do_estagio >= num_lotes:
                    break

        def run_gpu():
            print(
                f"\n[INFO] INICIANDO Fila de GPU: {len(tarefas_gpu)} modelos na base..."
            )
            with ProcessPoolExecutor(max_workers=1) as executor:
                while not queue_gpu.empty():
                    try:
                        cp, modelos = queue_gpu.get_nowait()
                    except queue.Empty:
                        break

                    future = executor.submit(
                        worker_treinar_lote, cp, modelos, global_permutation
                    )
                    processar_resultados_futures([future], "GPU-NATIVE", 1)
            print("[INFO] Fila de GPU finalizada.")

        def run_cpu():
            print(
                f"\n[INFO] INICIANDO Fila de CPU: {len(tarefas_cpu)} modelos nativos..."
            )
            with ProcessPoolExecutor(max_workers=3) as executor:
                cpu_futures = [
                    executor.submit(
                        worker_treinar_lote, cp, modelos, global_permutation
                    )
                    for cp, modelos in lotes_cpu_dict.items()
                ]
                processar_resultados_futures(
                    cpu_futures, "CPU-NATIVE", len(lotes_cpu_dict)
                )
            print(
                "[INFO] Fase nativa da CPU finalizada. Iniciando CPU-STEAL (Roubo de Carga)..."
            )

            # CPU Steal Phase
            lote_roubado = 1
            while not queue_gpu.empty():
                try:
                    cp, modelos = queue_gpu.get_nowait()
                except queue.Empty:
                    break

                print(
                    f"\n[CPU-STEAL LOTE {lote_roubado}] Carregando dataset e atacando {len(modelos)} modelos na memória RAM..."
                )
                X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = (
                    joblib.load(cp)
                )
                fs_cache = {}

                with ThreadPoolExecutor(max_workers=13) as steal_executor:
                    steal_futures = [
                        steal_executor.submit(
                            worker_single_model,
                            X_train,
                            y_train,
                            X_val,
                            y_val,
                            X_test,
                            y_test,
                            feature_cols,
                            model_name,
                            params,
                            exp_id,
                            global_permutation,
                            fs_cache,
                            force_cpu=True,
                        )
                        for model_name, params, exp_id in modelos
                    ]
                    processar_resultados_futures(
                        steal_futures, "CPU-STEAL", len(steal_futures)
                    )
                lote_roubado += 1

            print("[INFO] Steal Phase concluída. CPU ociosa.")

        t_gpu = threading.Thread(target=run_gpu)
        t_cpu = threading.Thread(target=run_cpu)

        t_gpu.start()
        t_cpu.start()

        t_gpu.join()
        t_cpu.join()

        parallel_time = time.time() - parallel_start
        for eid in exp_ids_to_update:
            db_manager.update_experiment_time(eid, parallel_time)

        print(
            f"[OK] {modelos_treinados} modelos computados assincronamente em {parallel_time:.1f}s."
        )

    # FECHAMENTO DA BATERIA GLOBAL
    battery_total_time = time.time() - battery_start_time
    db_manager.update_battery_time(battery_id, battery_total_time)
    print(
        f"\n[SUCESSO ABSOLUTO] Bateria Mãe (ID: {battery_id}) encerrada com sucesso total em {battery_total_time:.1f} segundos!"
    )


if __name__ == "__main__":
    if len(sys.argv) > 1:
        caminho_json = sys.argv[1]
    else:
        # Carregamento autônomo sem necessidade de passagem de parâmetro
        caminho_json = os.path.join(os.path.dirname(__file__), "config_experiment_bateria03.json")
        print(f"[SETUP] Nenhum parâmetro fornecido. Carregando padrão: {caminho_json}")

    executar_bateria_teste(caminho_json)
