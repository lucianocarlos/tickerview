import os
import json
import numpy as np
import pandas as pd
import warnings
from datetime import datetime

# Modelos do Scikit-Learn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.inspection import permutation_importance

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def gerar_target(df, strategy, config_target):
    """
    Gera a variável alvo (Y) de forma determinística.
    """
    df = df.copy()
    df = df.sort_values(["ticker", "Date"]).reset_index(drop=True)

    horizon = config_target.get("horizon_days", 10)
    threshold = config_target.get("threshold", 0.02)

    if strategy == "outperformance":
        df["Future_Return"] = df.groupby("ticker")["Close"].shift(-horizon) / df["Close"] - 1
        benchmark_return = df.groupby("Date")["Future_Return"].transform("median")
        df["Excess_Future_Return"] = df["Future_Return"] - benchmark_return
        df["target"] = (df["Excess_Future_Return"] >= threshold).astype(float)
        df.loc[df["Future_Return"].isna(), "target"] = np.nan
        cols_to_drop = ["Future_Return", "Excess_Future_Return"]

    elif strategy == "volatility_regime":
        df["Future_Vol"] = df.groupby("ticker")["Retorno_Diario"].transform(
            lambda x: x.rolling(window=horizon).std().shift(-horizon) * np.sqrt(252)
        )
        threshold_vol = df.groupby("ticker")["Volatilidade_21d"].transform(lambda x: x.quantile(0.70))
        df["target"] = (df["Future_Vol"] > threshold_vol).astype(float)
        df.loc[df["Future_Vol"].isna(), "target"] = np.nan
        cols_to_drop = ["Future_Vol"]

    elif strategy == "directional_regime":
        df["Future_Return"] = df.groupby("ticker")["Close"].shift(-horizon) / df["Close"] - 1
        df["target"] = 1.0  # Flat
        df.loc[df["Future_Return"] > threshold, "target"] = 2.0  # Bull
        df.loc[df["Future_Return"] < -threshold, "target"] = 0.0  # Bear
        df.loc[df["Future_Return"].isna(), "target"] = np.nan
        cols_to_drop = ["Future_Return"]

    elif strategy == "fundamental_turnaround":
        if "Lucro_Prejuizo_Periodo" not in df.columns:
            raise ValueError("A coluna 'Lucro_Prejuizo_Periodo' não está no dataset.")
        df["Future_Lucro"] = df.groupby("ticker")["Lucro_Prejuizo_Periodo"].shift(-horizon)
        cond_prejuizo_hoje = df["Lucro_Prejuizo_Periodo"] < 0
        cond_lucro_futuro = df["Future_Lucro"] > 0
        df["target"] = (cond_prejuizo_hoje & cond_lucro_futuro).astype(float)
        df.loc[df["Future_Lucro"].isna(), "target"] = np.nan
        cols_to_drop = ["Future_Lucro"]

    elif strategy == "drawdown_risk":
        df["Future_Min_Close"] = df.groupby("ticker")["Close"].transform(
            lambda x: x.rolling(window=horizon).min().shift(-horizon)
        )
        df["Drawdown_Futuro"] = (df["Future_Min_Close"] / df["Close"]) - 1
        threshold_dd = -0.20
        df["target"] = (df["Drawdown_Futuro"] < threshold_dd).astype(float)
        df.loc[df["Future_Min_Close"].isna(), "target"] = np.nan
        cols_to_drop = ["Future_Min_Close", "Drawdown_Futuro"]

    elif strategy == "dividend_trap_regime":
        cond_div = df["Dividends Payable"] > 0
        def get_min_future(x):
            import pandas as pd
            fwd = pd.concat([x.shift(-i) for i in range(1, horizon + 1)], axis=1)
            return fwd.min(axis=1)
        df["Future_Min_Close"] = df.groupby("ticker")["Close"].transform(get_min_future)
        df["Future_Drawdown"] = (df["Future_Min_Close"] - df["Close"]) / df["Close"].replace(0, 1)
        cond_crash = df["Future_Drawdown"] < -0.10
        df["target"] = (cond_div & cond_crash).astype(int)
        df.loc[df["Future_Min_Close"].isna(), "target"] = np.nan
        cols_to_drop = ["Future_Min_Close", "Future_Drawdown"]
    else:
        raise ValueError(f"Estratégia de target desconhecida: {strategy}")

    df = df.dropna(subset=["target"]).reset_index(drop=True)
    df["target"] = df["target"].astype(int)
    return df, cols_to_drop


def split_temporal(df, split_config, horizon_days=10):
    """Particionamento cronológico com Purging/Embargo Anti-Leakage."""
    # Garante a ordenação temporal estrita
    df = df.sort_values("Date").reset_index(drop=True)
    
    # Define o tamanho do Gap de Segurança em dias do calendário
    # O gap precisa ser pelo menos o horizonte do alvo (para não vazar target)
    # ou 21 dias (para expurgar a correlação da volatilidade 21d).
    gap_days = max(horizon_days, 21)
    
    method = split_config.get("method", "temporal_holdout")
    
    # Achar as datas limites ao invés de apenas índices para garantir 
    # o gap baseado no calendário real, evitando cortes no meio de um dia
    datas_unicas = df["Date"].sort_values().unique()
    total_dias = len(datas_unicas)
    
    if method == "temporal_holdout":
        train_ratio = split_config.get("train_ratio", 0.70)
        val_ratio = split_config.get("val_ratio", 0.15)
        
        idx_dia_train_end = int(total_dias * train_ratio)
        idx_dia_val_end = int(total_dias * (train_ratio + val_ratio))
        
        data_train_end = pd.to_datetime(datas_unicas[idx_dia_train_end])
        data_val_end = pd.to_datetime(datas_unicas[idx_dia_val_end])
        
        # Purging: Cortar o gap_days do fim do Treino e do fim da Validação
        data_train_purged = data_train_end - pd.Timedelta(days=gap_days)
        data_val_purged = data_val_end - pd.Timedelta(days=gap_days)

        df["Date_dt"] = pd.to_datetime(df["Date"])
        
        df_train = df[df["Date_dt"] <= data_train_purged].copy()
        df_val = df[(df["Date_dt"] > data_train_end) & (df["Date_dt"] <= data_val_purged)].copy()
        df_test = df[df["Date_dt"] > data_val_end].copy()
        
        df_train.drop(columns=["Date_dt"], inplace=True)
        df_val.drop(columns=["Date_dt"], inplace=True)
        df_test.drop(columns=["Date_dt"], inplace=True)
        
    elif method == "tsp":
        n_splits = split_config.get("n_splits", 5)
        block_size = total_dias // n_splits
        
        idx_dia_val_end = total_dias - block_size
        idx_dia_train_end = idx_dia_val_end - block_size
        
        data_train_end = pd.to_datetime(datas_unicas[idx_dia_train_end])
        data_val_end = pd.to_datetime(datas_unicas[idx_dia_val_end])
        
        # Purging: Cortar o gap_days do fim do Treino e do fim da Validação
        data_train_purged = data_train_end - pd.Timedelta(days=gap_days)
        data_val_purged = data_val_end - pd.Timedelta(days=gap_days)

        df["Date_dt"] = pd.to_datetime(df["Date"])
        
        df_train = df[df["Date_dt"] <= data_train_purged].copy()
        df_val = df[(df["Date_dt"] > data_train_end) & (df["Date_dt"] <= data_val_purged)].copy()
        df_test = df[df["Date_dt"] > data_val_end].copy()
        
        df_train.drop(columns=["Date_dt"], inplace=True)
        df_val.drop(columns=["Date_dt"], inplace=True)
        df_test.drop(columns=["Date_dt"], inplace=True)
        
    return df_train, df_val, df_test


def imputar_nulos_granular(df_train, df_val, df_test, feature_cols, imp_dict):
    """
    Roteia a imputação de forma granular baseada no tipo da coluna (Finanças, Estáticos, Balanço_YF).
    Aprendizado (medianas) é feito estritamente no df_train para evitar Data Leakage.
    """
    colunas_financas = ['Ativo_Total', 'Caixa_Equivalentes', 'Emprestimos_Curto_Prazo', 'Emprestimos_Longo_Prazo', 'Patrimonio_Liquido', 'EBIT_Operacional', 'Lucro_Prejuizo_Periodo', 'Divida_Total']
    colunas_estaticos = ['beta', 'marketCap', 'trailingPE']
    
    train_imp = df_train.copy()
    val_imp = df_val.copy()
    test_imp = df_test.copy()
    
    if isinstance(imp_dict, str):
        if imp_dict == "drop_all":
            train_imp.dropna(subset=feature_cols, inplace=True)
            val_imp.dropna(subset=feature_cols, inplace=True)
            test_imp.dropna(subset=feature_cols, inplace=True)
            return train_imp, val_imp, test_imp
        else:
            imp_dict = {
                "financas": imp_dict,
                "estaticos": imp_dict,
                "balanco_yf": imp_dict
            }
    
    # --- FINANÇAS ---
    strat_financas = imp_dict.get("financas", "fill_zero")
    cols_f = [c for c in colunas_financas if c in feature_cols]
    if strat_financas == "fill_zero":
        train_imp[cols_f] = train_imp[cols_f].fillna(0)
        val_imp[cols_f] = val_imp[cols_f].fillna(0)
        test_imp[cols_f] = test_imp[cols_f].fillna(0)
    elif strat_financas == "sector_median":
        medians = train_imp.groupby("sector")[cols_f].median()
        for s in medians.index:
            for c in cols_f:
                train_imp.loc[(train_imp["sector"] == s) & (train_imp[c].isna()), c] = medians.loc[s, c]
                val_imp.loc[(val_imp["sector"] == s) & (val_imp[c].isna()), c] = medians.loc[s, c]
                test_imp.loc[(test_imp["sector"] == s) & (test_imp[c].isna()), c] = medians.loc[s, c]
        global_medians = train_imp[cols_f].median()
        train_imp[cols_f] = train_imp[cols_f].fillna(global_medians)
        val_imp[cols_f] = val_imp[cols_f].fillna(global_medians)
        test_imp[cols_f] = test_imp[cols_f].fillna(global_medians)
    else:
        # Fallback de Segurança Anti-Crash
        train_imp[cols_f] = train_imp[cols_f].fillna(0)
        val_imp[cols_f] = val_imp[cols_f].fillna(0)
        test_imp[cols_f] = test_imp[cols_f].fillna(0)

    # --- ESTATICOS ---
    strat_estaticos = imp_dict.get("estaticos", "sector_median")
    cols_e = [c for c in colunas_estaticos if c in feature_cols]
    if strat_estaticos == "fill_zero":
        train_imp[cols_e] = train_imp[cols_e].fillna(0)
        val_imp[cols_e] = val_imp[cols_e].fillna(0)
        test_imp[cols_e] = test_imp[cols_e].fillna(0)
    elif strat_estaticos == "global_median":
        global_medians = train_imp[cols_e].median()
        train_imp[cols_e] = train_imp[cols_e].fillna(global_medians)
        val_imp[cols_e] = val_imp[cols_e].fillna(global_medians)
        test_imp[cols_e] = test_imp[cols_e].fillna(global_medians)
    elif strat_estaticos == "sector_median":
        medians = train_imp.groupby("sector")[cols_e].median()
        for s in medians.index:
            for c in cols_e:
                train_imp.loc[(train_imp["sector"] == s) & (train_imp[c].isna()), c] = medians.loc[s, c]
                val_imp.loc[(val_imp["sector"] == s) & (val_imp[c].isna()), c] = medians.loc[s, c]
                test_imp.loc[(test_imp["sector"] == s) & (test_imp[c].isna()), c] = medians.loc[s, c]
        global_medians = train_imp[cols_e].median()
        train_imp[cols_e] = train_imp[cols_e].fillna(global_medians)
        val_imp[cols_e] = val_imp[cols_e].fillna(global_medians)
        test_imp[cols_e] = test_imp[cols_e].fillna(global_medians)
    else:
        # Fallback de Segurança Anti-Crash
        train_imp[cols_e] = train_imp[cols_e].fillna(0)
        val_imp[cols_e] = val_imp[cols_e].fillna(0)
        test_imp[cols_e] = test_imp[cols_e].fillna(0)
        
    # --- BALANÇO YF (Restante Numérico) ---
    strat_yf = imp_dict.get("balanco_yf", "fill_zero")
    cols_yf = [c for c in feature_cols if c not in cols_f + cols_e]
    if strat_yf == "fill_zero":
        train_imp[cols_yf] = train_imp[cols_yf].fillna(0)
        val_imp[cols_yf] = val_imp[cols_yf].fillna(0)
        test_imp[cols_yf] = test_imp[cols_yf].fillna(0)
    elif strat_yf == "sector_median":
        medians = train_imp.groupby("sector")[cols_yf].median()
        for s in medians.index:
            for c in cols_yf:
                train_imp.loc[(train_imp["sector"] == s) & (train_imp[c].isna()), c] = medians.loc[s, c]
                val_imp.loc[(val_imp["sector"] == s) & (val_imp[c].isna()), c] = medians.loc[s, c]
                test_imp.loc[(test_imp["sector"] == s) & (test_imp[c].isna()), c] = medians.loc[s, c]
        global_medians = train_imp[cols_yf].median()
        train_imp[cols_yf] = train_imp[cols_yf].fillna(global_medians)
        val_imp[cols_yf] = val_imp[cols_yf].fillna(global_medians)
        test_imp[cols_yf] = test_imp[cols_yf].fillna(global_medians)
    else:
        # Fallback de Segurança Anti-Crash
        train_imp[cols_yf] = train_imp[cols_yf].fillna(0)
        val_imp[cols_yf] = val_imp[cols_yf].fillna(0)
        test_imp[cols_yf] = test_imp[cols_yf].fillna(0)
        
    return train_imp, val_imp, test_imp


def treinar_e_avaliar_modelo(df_raw, target_strategy, target_definition, split_config, 
                             inf_handling_strategy, imputation_strategy, outlier_handling, scaling_method, model_name, hparams,
                             calculate_permutation_importance=True):
    """
    Motor central purificado. Recebe dados brutos, hiperparâmetros e retorna métricas e XAI.
    Sem chamadas de salvamento de arquivos. Tudo é retornado em memória.
    """
    # 1. Target
    df_target, cols_to_drop = gerar_target(df_raw, target_strategy, target_definition)

    # 2. Separar Features Numéricas
    metadata_cols = ["Date", "ticker", "target", "sector", "industry", "tamanho_categoria", "Open", "High", "Low", "Close", "Volume"] + cols_to_drop
    cols_to_exclude = [col for col in metadata_cols if col in df_target.columns]
    feature_cols = [col for col in df_target.columns if col not in cols_to_exclude and pd.api.types.is_numeric_dtype(df_target[col])]

    # 3. Tratamento de Infinitos (Seguro fazer antes do Split, pois é Cell-wise)
    if inf_handling_strategy == "replace_nan":
        df_target[feature_cols] = df_target[feature_cols].replace([np.inf, -np.inf], np.nan)
    elif inf_handling_strategy == "drop_row":
        df_target = df_target[~np.isinf(df_target[feature_cols]).any(axis=1)].reset_index(drop=True)

    # 4. Split Temporal (Criando as partições independentes)
    horizon_days = target_definition.get("horizon_days", 10)
    df_train, df_val, df_test = split_temporal(df_target, split_config, horizon_days=horizon_days)
    
    # NOVO: Saneamento Populacional Blindado (Anti-Data Leakage)
    # 5. Imputação de Nulos (Aprende apenas no Treino com Granularidade)
    # Se receber uma string por compatibilidade legada, transforma em dict
    if isinstance(imputation_strategy, str):
        imputation_strategy = {
            "financas": imputation_strategy,
            "estaticos": imputation_strategy,
            "balanco_yf": imputation_strategy
        }

    df_train, df_val, df_test = imputar_nulos_granular(
        df_train, df_val, df_test, feature_cols, imputation_strategy
    )

    # 5. Tratamento de Outliers (Aprende apenas no Treino)
    if outlier_handling == "clip_99_1":
        # Extrai os percentis EXCLUSIVAMENTE do Treino
        lower = df_train[feature_cols].quantile(0.01)
        upper = df_train[feature_cols].quantile(0.99)
        # Corta todos usando o limite do Treino
        df_train[feature_cols] = df_train[feature_cols].clip(lower=lower, upper=upper, axis=1)
        df_val[feature_cols] = df_val[feature_cols].clip(lower=lower, upper=upper, axis=1)
        df_test[feature_cols] = df_test[feature_cols].clip(lower=lower, upper=upper, axis=1)

    # Separa Features e Target
    X_train, y_train = df_train[feature_cols].copy(), df_train["target"]
    X_val, y_val = df_val[feature_cols].copy(), df_val["target"]
    X_test, y_test = df_test[feature_cols].copy(), df_test["target"]

    # 7. Scaling Global (Aprende apenas no Treino - Já estava blindado!)
    if scaling_method != "none" and len(feature_cols) > 0:
        if scaling_method == "standard": scaler = StandardScaler()
        elif scaling_method == "minmax": scaler = MinMaxScaler()
        elif scaling_method == "robust": scaler = RobustScaler()
        X_train[feature_cols] = scaler.fit_transform(X_train[feature_cols])
        X_val[feature_cols] = scaler.transform(X_val[feature_cols])
        X_test[feature_cols] = scaler.transform(X_test[feature_cols])

    # 5. Feature Selection
    feature_selection_top_k = hparams.get("feature_selection_top_k", None)
    clean_hparams = {k: v for k, v in hparams.items() if not k.startswith("_") and k != "feature_selection_top_k"}

    if feature_selection_top_k is not None and feature_selection_top_k < len(feature_cols):
        selector = SelectFromModel(
            DecisionTreeClassifier(random_state=42, max_depth=10),
            max_features=feature_selection_top_k, threshold=-np.inf
        )
        selector.fit(X_train[feature_cols], y_train)
        feature_cols = np.array(feature_cols)[selector.get_support()].tolist()
        X_train, X_val, X_test = X_train[feature_cols], X_val[feature_cols], X_test[feature_cols]

    # 6. Instanciar Modelo (Reprodutibilidade: Random State 42 sempre que possível)
    if 'random_state' not in clean_hparams and model_name not in ['knn', 'voting_classifier']:
        clean_hparams['random_state'] = 42

    if model_name == "decision_tree": model = DecisionTreeClassifier(**clean_hparams)
    elif model_name == "random_forest": model = RandomForestClassifier(n_jobs=1, **clean_hparams)
    elif model_name == "knn": model = KNeighborsClassifier(n_jobs=1, **clean_hparams)
    elif model_name == "mlp": model = MLPClassifier(**clean_hparams)
    elif model_name == "logistic_regression": model = LogisticRegression(n_jobs=1, **clean_hparams)
    elif model_name == "xgboost":
        if XGBOOST_AVAILABLE: model = XGBClassifier(n_jobs=1, **clean_hparams)
        else: model = RandomForestClassifier(n_jobs=1, **clean_hparams)
    elif model_name == "voting_classifier":
        # Usamos os 4 campeões que você documentou no seu mestrado/doutorado
        estimators = [
            ('rf', RandomForestClassifier(n_estimators=80, max_depth=8, class_weight='balanced', random_state=42, n_jobs=1)),
            ('xgb', XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, n_jobs=1) if XGBOOST_AVAILABLE else RandomForestClassifier(random_state=42, n_jobs=1)),
            ('mlp', MLPClassifier(hidden_layer_sizes=(100,), activation='relu', max_iter=1000, random_state=42)),
            ('lg', LogisticRegression(C=1.0, penalty='l2', random_state=42, n_jobs=1))
        ]
        model = VotingClassifier(estimators=estimators, voting='hard', **clean_hparams)
    else:
        raise ValueError(f"Modelo desconhecido: {model_name}")

    # 7. Treinamento
    model.fit(X_train, y_train)

    # 8. Métricas
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)
    
    metrics = {
        "val_accuracy": accuracy_score(y_val, y_val_pred),
        "val_f1_macro": f1_score(y_val, y_val_pred, average="macro", zero_division=0),
        "test_accuracy": accuracy_score(y_test, y_test_pred),
        "test_f1_macro": f1_score(y_test, y_test_pred, average="macro", zero_division=0),
        "test_f1_weighted": f1_score(y_test, y_test_pred, average="weighted", zero_division=0),
        "test_precision_macro": precision_score(y_test, y_test_pred, average="macro", zero_division=0),
        "test_recall_macro": recall_score(y_test, y_test_pred, average="macro", zero_division=0)
    }
    confusion_mat = confusion_matrix(y_test, y_test_pred).tolist()

    # 9. Extração Universal de Explicabilidade (XAI)
    feature_importances = {}
    importance_type = "none"
    
    # Tentativa 1: Modelos baseados em Árvore (Entropy/Gini)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        importance_type = "entropy"
        indices = np.argsort(importances)[::-1]
        for idx in indices[:20]: # Salva as top 20 para economizar banco, ou podemos salvar tudo.
            if importances[idx] > 0:
                feature_importances[feature_cols[idx]] = float(importances[idx])
                
    # Tentativa 2: Modelos Lineares (Coeficientes)
    elif hasattr(model, "coef_"):
        # Pegamos a magnitude absoluta dos coeficientes (para classificação multi-classe, tiramos a média absoluta)
        coefs = np.mean(np.abs(model.coef_), axis=0) if len(model.coef_.shape) > 1 else np.abs(model.coef_[0])
        importance_type = "coefficient"
        indices = np.argsort(coefs)[::-1]
        for idx in indices[:20]:
            if coefs[idx] > 0:
                feature_importances[feature_cols[idx]] = float(coefs[idx])
                
    # Tentativa 3: Caixas-Pretas (Permutation Importance do Scikit-Learn)
    else:
        if calculate_permutation_importance:
            importance_type = "permutation"
            # Usamos o dataset de Validação para calcular a Permutation Importance (boa prática)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = permutation_importance(model, X_val, y_val, n_repeats=5, random_state=42, n_jobs=1)
                importances = result.importances_mean
                indices = np.argsort(importances)[::-1]
                for idx in indices[:20]:
                    if importances[idx] > 0:
                        feature_importances[feature_cols[idx]] = float(importances[idx])
        else:
            importance_type = "skipped"

    return metrics, confusion_mat, feature_importances, importance_type
