import os
import sys
import json
import time
import pandas as pd
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed

# Importações locais do MLOps
import db_manager
from classification import classifier

def worker_treinar_modelo(df_raw, strategy, target_def, split_config, 
                          inf_strat, imp_strategy, out_method, scale_method, 
                          model_name, params, exp_id, calc_permutation):
    
    model_start = time.time()
    try:
        metrics, conf_mat, feature_importances, importance_type = classifier.treinar_e_avaliar_modelo(
            df_raw=df_raw,
            target_strategy=strategy,
            target_definition=target_def,
            split_config=split_config,
            inf_handling_strategy=inf_strat,
            imputation_strategy=imp_strategy,
            outlier_handling=out_method,
            scaling_method=scale_method,
            model_name=model_name,
            hparams=params,
            calculate_permutation_importance=calc_permutation
        )
        model_time = time.time() - model_start
        return {
            "status": "success",
            "model_name": model_name,
            "params": params,
            "metrics": metrics,
            "conf_mat": conf_mat,
            "feature_importances": feature_importances,
            "importance_type": importance_type,
            "model_time": model_time
        }
    except Exception as e:
        return {
            "status": "error",
            "model_name": model_name,
            "params": params,
            "error_msg": str(e)
        }

def executar_bateria_teste(config_path):
    battery_start_time = time.time()

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # NOVO: Chave mestra de performance
    global_permutation = config.get("calculate_permutation_importance", False)

    # O nome do Datalake vem explicitamente de dentro do JSON (ou usa o nome do arquivo se não existir)
    file_name_fallback = os.path.basename(config_path).replace(".json", "")
    battery_name = config.get("datalake_name", file_name_fallback)
    
    dynamic_db_path = os.path.join("data", "datalake", battery_name, "datalake.db")
    db_manager.set_db_path(dynamic_db_path)

    # Verifica se já existe um checkpoint Pai (a própria Bateria)
    battery_id = db_manager.get_or_create_battery(battery_name, config)
    print("="*50)
    print("      MLOps 2.0 - SQLite Relacional Datalake      ")
    print("="*50)
    print(f"[BATTERY] Bateria Global Iniciada/Retomada! ID: {battery_id} | Name: {battery_name}\n")

    dataset_vs = config.get("datasets", ["dataset003"])
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    for dataset_v in dataset_vs:
        print(f"\n[INFO] Iniciando Pipeline para o Dataset: {dataset_v}")
        
        # 1. Carregar Dados Brutos e Info do Dataset
        dataset_path = os.path.join(project_root, "data", "dataset", dataset_v, "mestre.parquet")
        info_dataset_path = os.path.join(project_root, "data", "dataset", dataset_v, f"info_{dataset_v}.json")
        
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
            generation_parameters=gen_params
        )
        print(f"       -> Dataset cadastrado/recuperado no Banco! ID: {dataset_id}")

        split_strategies = config.get("split_strategies", [{"method": "temporal_holdout", "train_ratio": 0.7}])
        target_strategies = config.get("target_strategies", ["outperformance"])
        inf_strategies = config.get("inf_handling_strategies", ["replace_nan"])
        imputation_strategies = config.get("imputation_strategies", ["fill_zero"])
        outlier_methods = config.get("outlier_handling_methods", ["none"])
        scaling_methods = config.get("scaling_methods", ["none"])
        models_config = config.get("models", {})
        target_def = config.get("target_definition", {"horizon_days": 10, "threshold": 0.02})

        for strategy in target_strategies:
            for split_config in split_strategies:
             for inf_strat in inf_strategies:
              for imp_strategy in imputation_strategies:
                for out_method in outlier_methods:
                  for scale_method in scaling_methods:
                    
                    print(f"\n[EXPERIMENT] Target: {strategy} | Inf: {inf_strat} | Imp: {imp_strategy} | Out: {out_method} | Scale: {scale_method}")
                    
                    # Empacotando toda a inteligência relacional num único dicionário JSON
                    experiment_config = {
                        "target_definition": target_def,
                        "split_config": split_config,
                        "inf_handling_strategy": inf_strat,
                        "imputation_strategy": imp_strategy,
                        "outlier_handling": out_method,
                        "scaling_method": scale_method,
                        "grid_search_config": models_config
                    }
                    
                    # 4. Criar ou Recuperar o Experimento (Filho da Bateria) no Banco
                    exp_id = db_manager.get_or_create_experiment(
                        battery_id=battery_id,
                        dataset_id=dataset_id,
                        task_type="classification",
                        target_strategy=strategy,
                        experiment_config=experiment_config
                    )
                    
                    exp_start_time = time.time()
                    modelos_treinados = 0
                    
                    # 5. Mapear Modelos Pendentes (Filtro Checkpoint)
                    tasks = []
                    for model_name, param_grid in models_config.items():
                        if model_name.startswith("_"): continue
                        
                        clean_param_grid = {k: v for k, v in param_grid.items() if not k.startswith("_")}
                        grid = list(ParameterGrid(clean_param_grid))
                        
                        for params in grid:
                            # Filtro Mágico: O Banco proíbe retrabalho
                            if not db_manager.model_exists(exp_id, model_name, params):
                                tasks.append((model_name, params))
                            else:
                                print(f"   -> [SKIPPED] {model_name} já treinado nesta arquitetura.")
                    
                    if not tasks:
                        print("   -> Todos os modelos já foram treinados neste experimento (100% Retomado). Pulando...")
                        continue
                        
                    print(f"   -> Disparando {len(tasks)} treinamentos pesados para a CPU (Multiprocessing 100%)...")
                    
                    # 6. Grid Search Paralelo (CPU a 100% devolvendo via Generator em tempo real)
                    resultados = Parallel(n_jobs=-1, return_as="generator")(
                        delayed(worker_treinar_modelo)(
                            df_raw, strategy, target_def, split_config, 
                            inf_strat, imp_strategy, out_method, scale_method, 
                            m_name, m_params, exp_id, global_permutation
                        ) for m_name, m_params in tasks
                    )
                    
                    # 7. Escrita Segura e Sequencial no SQLite
                    for res in resultados:
                        if res["status"] == "success":
                            db_manager.save_model_results(
                                experiment_id=exp_id,
                                model_name=res["model_name"],
                                hyperparameters=res["params"],
                                exec_time_sec=res["model_time"],
                                metrics_class=res["metrics"],
                                confusion_matrix=res["conf_mat"],
                                feature_importances=res["feature_importances"],
                                importance_type=res["importance_type"]
                            )
                            modelos_treinados += 1
                            print(f"   -> [DB INSERT] {res['model_name']} salvo. F1-Macro: {res['metrics']['test_f1_macro']:.4f} | Tempo: {res['model_time']:.1f}s")
                        elif res["status"] == "error":
                            print(f"   -> [ERRO FATAL] Falha no {res['model_name']}: {res['error_msg']}")
                            
                    # 7. Atualizar Tempo Total do Experimento
                    exp_total_time = time.time() - exp_start_time
                    db_manager.update_experiment_time(exp_id, exp_total_time)
                    print(f"[OK] Experimento {exp_id} finalizado em {exp_total_time:.1f}s. ({modelos_treinados} modelos computados)")

    # FECHAMENTO DA BATERIA GLOBAL
    battery_total_time = time.time() - battery_start_time
    db_manager.update_battery_time(battery_id, battery_total_time)
    print(f"\n[SUCESSO ABSOLUTO] Bateria Mãe (ID: {battery_id}) encerrada com sucesso total em {battery_total_time:.1f} segundos!")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        caminho_json = sys.argv[1]
    else:
        # Carregamento autônomo sem necessidade de passagem de parâmetro
        caminho_json = os.path.join(os.path.dirname(__file__), "config_experiment.json")
        print(f"[SETUP] Nenhum parâmetro fornecido. Carregando padrão: {caminho_json}")
        
    executar_bateria_teste(caminho_json)
