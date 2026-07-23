import numpy as np
import pandas as pd
import warnings

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.inspection import permutation_importance

# Modelos do Scikit-Learn mantidos para legado/feature selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import threading

# Lock global para isolamento de threads no Feature Selection Cache
fs_global_lock = threading.Lock()


# Suprimir avisos irrelevantes de performance do Pandas e do XGBoost
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

# Helper para checar device globalmente
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"TickerView Engine Iniciado. Dispositivo principal de processamento: {DEVICE}")

# =========================================================================
# WRAPPERS PYTORCH (Substituindo Scikit-Learn e enviando para VRAM)
# =========================================================================


class PyTorchMLP(nn.Module):
    def __init__(
        self, input_size, hidden_sizes=(100,), num_classes=2, activation="relu", dropout_p=0.3
    ):
        super().__init__()
        layers = []
        in_size = input_size

        # Mapeamento do Scikit-Learn para o PyTorch
        if activation == "tanh":
            act_layer = nn.Tanh
        elif activation == "logistic":
            act_layer = nn.Sigmoid
        else:
            act_layer = nn.ReLU

        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            if h > 1:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act_layer())
            if dropout_p > 0:
                layers.append(nn.Dropout(p=dropout_p))
            in_size = h
        layers.append(nn.Linear(in_size, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class SklearnPyTorchMLPWrapper:
    """Wrapper para fazer o PyTorch agir como um modelo do Scikit-Learn"""

    def __init__(
        self,
        hidden_layer_sizes=(100,),
        activation="relu",
        max_iter=1000,
        learning_rate_init=0.001,
        weight_decay=0.0,
        dropout_p=0.3,
        class_weight=None,
        patience=15,
        tol=1e-4,
        random_state=42,
        device=None,
        **kwargs,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.max_iter = max_iter
        self.lr = learning_rate_init
        self.weight_decay = weight_decay
        self.dropout_p = dropout_p
        self.class_weight = class_weight
        self.patience = patience
        self.tol = tol
        self.random_state = random_state
        self.device = device if device is not None else DEVICE
        self.model = None
        self.classes_ = None
        self.n_iter_ = 0

    def fit(self, X, y):
        torch.manual_seed(self.random_state)

        # Converte para Tensores e envia para a GPU
        X_t = torch.tensor(
            X.values if isinstance(X, pd.DataFrame) else X, dtype=torch.float32
        ).to(self.device)
        y_t = torch.tensor(
            y.values if isinstance(y, pd.Series) else y, dtype=torch.int64
        ).to(self.device)

        self.classes_ = np.unique(y_t.cpu().numpy())
        num_classes = len(self.classes_)
        input_size = X_t.shape[1]

        self.model = PyTorchMLP(
            input_size, self.hidden_layer_sizes, num_classes, self.activation, self.dropout_p
        ).to(self.device)

        criterion_weight = None
        if self.class_weight == "balanced":
            class_counts = torch.bincount(y_t)
            total_samples = len(y_t)
            weights = total_samples / (num_classes * class_counts.float())
            criterion_weight = weights.to(self.device)

        criterion = nn.CrossEntropyLoss(weight=criterion_weight)
        optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        self.model.train()
        best_loss = float('inf')
        patience_counter = 0
        self.n_iter_ = 0

        for epoch in range(self.max_iter):
            self.n_iter_ = epoch + 1
            optimizer.zero_grad()
            outputs = self.model(X_t)
            loss = criterion(outputs, y_t)
            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            if current_loss < best_loss - self.tol:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    break

        return self

    def predict(self, X):
        self.model.eval()
        X_t = torch.tensor(
            X.values if isinstance(X, pd.DataFrame) else X, dtype=torch.float32
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_t)
            _, predicted = torch.max(outputs.data, 1)
        return predicted.cpu().numpy()

    def score(self, X, y):
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))


class PyTorchKNNWrapper:
    """KNN 100% otimizado via Pytorch (MUITO MAIS RÁPIDO QUE CPU)"""

    def __init__(self, n_neighbors=5, metric="minkowski", weights="uniform", device=None):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.weights = weights
        self.device = device if device is not None else DEVICE
        self.p = 1.0 if metric == "manhattan" else 2.0  # minkowski/euclidean = 2.0
        self.X_train_t = None
        self.y_train_t = None
        self.num_classes = 2

    def fit(self, X, y):
        self.X_train_t = torch.tensor(
            X.values if isinstance(X, pd.DataFrame) else X, dtype=torch.float32
        ).to(self.device)
        self.y_train_t = torch.tensor(
            y.values if isinstance(y, pd.Series) else y, dtype=torch.int64
        ).to(self.device)
        self.num_classes = len(torch.unique(self.y_train_t))
        return self

    def predict(self, X):
        X_t = torch.tensor(
            X.values if isinstance(X, pd.DataFrame) else X, dtype=torch.float32
        ).to(self.device)

        batch_size = 1000
        predictions = []

        for i in range(0, X_t.shape[0], batch_size):
            X_batch = X_t[i : i + batch_size]

            # Cálculo de distância com métrica 'p'
            dists = torch.cdist(X_batch, self.X_train_t, p=self.p)
            topk_dists, topk_indices = torch.topk(
                dists, self.n_neighbors, dim=1, largest=False
            )
            topk_labels = self.y_train_t[topk_indices]

            # Sistema de Votação Otimizado via Tensor Scatter
            batch_sz = X_batch.shape[0]
            votes = torch.zeros(batch_sz, self.num_classes, device=self.device)

            if self.weights == "distance":
                w = 1.0 / (topk_dists + 1e-8)
            else:
                w = torch.ones_like(topk_dists)

            votes.scatter_add_(1, topk_labels, w)
            batch_preds = torch.argmax(votes, dim=1)

            predictions.append(batch_preds.cpu().numpy())

        return np.concatenate(predictions)

    def score(self, X, y):
        from sklearn.metrics import accuracy_score
        return accuracy_score(y, self.predict(X))


class SklearnPyTorchLogisticRegressionWrapper(SklearnPyTorchMLPWrapper):
    """Regressão Logística Mapeada (Rede Neural sem Hidden Layers e Adam com L2/Weight Decay)"""

    def __init__(self, C=1.0, penalty="l2", max_iter=1000, random_state=42, device=None, **kwargs):
        # Mapeamento do C (Inverso da Regularização Scikit-Learn) para Weight Decay no PyTorch
        wd = 1.0 / C if C > 0 else 0.0
        # A Regressão Logística usa 'logistic' (Sigmoid) internamente na formulação clássica,
        # mas como não há hidden layers, CrossEntropyLoss finaliza o trabalho. Passamos relu só por sintaxe.
        super().__init__(
            hidden_layer_sizes=(),
            activation="relu",
            max_iter=max_iter,
            learning_rate_init=0.01,
            weight_decay=wd,
            random_state=random_state,
            device=device,
            **kwargs,
        )


class GPUVotingClassifier:
    """Orquestrador Híbrido: Delega o predict para XGBoost e PyTorch na GPU e junta os votos ponderados."""

    def __init__(self, estimators, weights=None):
        self.estimators = estimators
        self.weights = weights

    def fit(self, X, y):
        for name, model in self.estimators:
            model.fit(X, y)
        return self

    def predict(self, X):
        all_preds = []
        for name, model in self.estimators:
            all_preds.append(model.predict(X))

        all_preds = np.vstack(all_preds)  # (n_estimators, n_samples)

        if self.weights is None:
            import scipy.stats
            majority_vote, _ = scipy.stats.mode(all_preds, axis=0, keepdims=False)
            return majority_vote

        # Votação ponderada
        classes = np.unique(all_preds)
        n_samples = all_preds.shape[1]
        weighted_votes = np.zeros((n_samples, len(classes)))

        for i in range(len(self.estimators)):
            for j, c in enumerate(classes):
                weighted_votes[:, j] += (all_preds[i] == c) * self.weights[i]

        return classes[np.argmax(weighted_votes, axis=1)]


try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


def gerar_target(df, strategy, config_target):
    """
    Gera a variável alvo (Y) de forma determinística com calibração adaptativa de volatilidade.
    """
    df = df.copy()
    df = df.sort_values(["ticker", "Date"]).reset_index(drop=True)

    horizon = config_target.get("horizon_days", 10)
    threshold = config_target.get("threshold", 0.02)
    use_vol_scaling = config_target.get("use_vol_scaling", True)

    # Cálculo da volatilidade normalizada no horizonte H se disponível
    has_vol = "Volatilidade_21d" in df.columns and use_vol_scaling
    if has_vol:
        # Volatilidade adaptativa ajustada pela raiz do horizonte de tempo (anualizado)
        scale_factor = np.sqrt(horizon / 252.0)
        vol_h = (df["Volatilidade_21d"] * scale_factor).clip(lower=0.01)
    else:
        vol_h = threshold

    if strategy == "outperformance":
        df["Future_Return"] = (
            df.groupby("ticker")["Close"].shift(-horizon) / df["Close"] - 1
        )
        benchmark_return = df.groupby("Date")["Future_Return"].transform("median")
        df["Excess_Future_Return"] = df["Future_Return"] - benchmark_return
        
        # Limiar adaptativo: garante que a outperformance seja estatisticamente significativa
        if has_vol:
            dynamic_thresh = np.maximum(threshold, vol_h * 0.5)
        else:
            dynamic_thresh = threshold

        df["target"] = (df["Excess_Future_Return"] >= dynamic_thresh).astype(float)
        df.loc[df["Future_Return"].isna(), "target"] = np.nan
        cols_to_drop = ["Future_Return", "Excess_Future_Return"]

    elif strategy == "volatility_regime":
        # Janela móvel de volatilidade no horizonte H
        df["Future_Vol"] = df.groupby("ticker")["Retorno_Diario"].transform(
            lambda x: x.rolling(window=max(horizon, 5), min_periods=3).std().shift(-horizon) * np.sqrt(252)
        )
        
        if has_vol:
            # Razão de expansão de volatilidade: detecta se a volatilidade futura expande > 20% sobre a média de 21d
            vol_expansion_ratio = df["Future_Vol"] / df["Volatilidade_21d"].replace(0, np.nan)
            ratio_thresh = config_target.get("expansion_ratio", 1.20)
            df["target"] = (vol_expansion_ratio > ratio_thresh).astype(float)
        else:
            threshold_vol = df.groupby("ticker")["Volatilidade_21d"].transform(
                lambda x: x.quantile(0.70)
            )
            df["target"] = (df["Future_Vol"] > threshold_vol).astype(float)

        df.loc[df["Future_Vol"].isna(), "target"] = np.nan
        cols_to_drop = ["Future_Vol"]

    elif strategy == "directional_regime":
        df["Future_Return"] = (
            df.groupby("ticker")["Close"].shift(-horizon) / df["Close"] - 1
        )
        
        # Limiar de transição de regime ajustado por ativo (evita colapso na classe Flat em papéis defensivos)
        if has_vol:
            dynamic_thresh = np.maximum(0.015, vol_h * 0.75)
        else:
            dynamic_thresh = threshold

        df["target"] = 1.0  # Flat / Neutro
        df.loc[df["Future_Return"] > dynamic_thresh, "target"] = 2.0  # Bull
        df.loc[df["Future_Return"] < -dynamic_thresh, "target"] = 0.0  # Bear
        df.loc[df["Future_Return"].isna(), "target"] = np.nan
        cols_to_drop = ["Future_Return"]

    elif strategy == "fundamental_turnaround":
        if "Lucro_Prejuizo_Periodo" not in df.columns:
            raise ValueError("A coluna 'Lucro_Prejuizo_Periodo' não está no dataset.")
        df["Future_Lucro"] = df.groupby("ticker")["Lucro_Prejuizo_Periodo"].shift(
            -horizon
        )
        cond_prejuizo_hoje = df["Lucro_Prejuizo_Periodo"] < 0
        cond_lucro_futuro = df["Future_Lucro"] > 0
        
        # Se houver EBIT, valida se também houve recuperação operacional
        if "EBIT_Operacional" in df.columns:
            df["Future_EBIT"] = df.groupby("ticker")["EBIT_Operacional"].shift(-horizon)
            cond_ebit_rec = df["Future_EBIT"] > df["EBIT_Operacional"]
            df["target"] = (cond_prejuizo_hoje & cond_lucro_futuro & cond_ebit_rec).astype(float)
            cols_to_drop = ["Future_Lucro", "Future_EBIT"]
        else:
            df["target"] = (cond_prejuizo_hoje & cond_lucro_futuro).astype(float)
            cols_to_drop = ["Future_Lucro"]

        df.loc[df["Future_Lucro"].isna(), "target"] = np.nan

    elif strategy == "drawdown_risk":
        df["Future_Min_Close"] = df.groupby("ticker")["Close"].transform(
            lambda x: x.rolling(window=horizon, min_periods=1).min().shift(-horizon)
        )
        df["Drawdown_Futuro"] = (df["Future_Min_Close"] / df["Close"]) - 1
        
        if has_vol:
            dynamic_dd = -np.maximum(abs(threshold), vol_h * 0.6)
        else:
            dynamic_dd = -abs(threshold)

        df["target"] = (df["Drawdown_Futuro"] < dynamic_dd).astype(float)
        df.loc[df["Future_Min_Close"].isna(), "target"] = np.nan
        cols_to_drop = ["Future_Min_Close", "Drawdown_Futuro"]

    elif strategy == "dividend_trap_regime":
        cond_div = df["Dividends Payable"] > 0

        def get_min_future(x):
            import pandas as pd
            fwd = pd.concat([x.shift(-i) for i in range(1, horizon + 1)], axis=1)
            return fwd.min(axis=1)

        df["Future_Min_Close"] = df.groupby("ticker")["Close"].transform(get_min_future)
        df["Future_Drawdown"] = (df["Future_Min_Close"] - df["Close"]) / df[
            "Close"
        ].replace(0, 1)

        if has_vol:
            dynamic_crash = -np.maximum(abs(threshold), vol_h * 0.7)
        else:
            dynamic_crash = -abs(threshold)

        cond_crash = df["Future_Drawdown"] < dynamic_crash
        df["target"] = (cond_div & cond_crash).astype(float)
        df.loc[df["Future_Min_Close"].isna(), "target"] = np.nan
        cols_to_drop = ["Future_Min_Close", "Future_Drawdown"]

    elif strategy == "triple_barrier":
        pt_sl_ratio = config_target.get("pt_sl_ratio", [1.0, 1.0])
        up_factor, down_factor = pt_sl_ratio[0], pt_sl_ratio[1]
        
        vol_barrier = vol_h if has_vol else threshold
        up_limit = up_factor * vol_barrier
        dn_limit = -down_factor * vol_barrier
        
        fwd_returns = pd.concat([(df.groupby("ticker")["Close"].shift(-h) / df["Close"]) - 1 for h in range(1, horizon + 1)], axis=1)
        
        hit_up = fwd_returns.ge(up_limit, axis=0).values
        hit_dn = fwd_returns.le(dn_limit, axis=0).values
        
        hit_up_idx = np.argmax(hit_up, axis=1)
        hit_dn_idx = np.argmax(hit_dn, axis=1)
        
        up_valid = hit_up[np.arange(len(hit_up)), hit_up_idx]
        dn_valid = hit_dn[np.arange(len(hit_dn)), hit_dn_idx]
        
        hit_up_idx = np.where(up_valid, hit_up_idx, 9999)
        hit_dn_idx = np.where(dn_valid, hit_dn_idx, 9999)
        
        target = np.full(len(df), np.nan)
        
        mask_up_first = (hit_up_idx < hit_dn_idx)
        target[mask_up_first] = 1.0
        
        mask_dn_first = (hit_dn_idx < hit_up_idx)
        target[mask_dn_first] = 0.0
        
        mask_simul = (hit_up_idx == hit_dn_idx) & (hit_up_idx != 9999)
        target[mask_simul] = 0.0
        
        mask_no_hit = (hit_up_idx == 9999) & (hit_dn_idx == 9999)
        final_ret = fwd_returns.iloc[:, -1].values
        target[mask_no_hit] = np.where(final_ret[mask_no_hit] > 0, 1.0, 0.0)
        
        df["target"] = target
        df.loc[fwd_returns.iloc[:, -1].isna(), "target"] = np.nan
        cols_to_drop = []

    elif strategy == "vol_adjusted_return":
        future_return = df.groupby("ticker")["Close"].shift(-horizon) / df["Close"] - 1
        vol_adjusted = future_return / vol_h
        dynamic_thresh = config_target.get("sharpe_threshold", 0.5)
        df["target"] = (vol_adjusted >= dynamic_thresh).astype(float)
        df.loc[future_return.isna(), "target"] = np.nan
        cols_to_drop = []

    elif strategy == "cross_sectional_quantiles":
        future_return = df.groupby("ticker")["Close"].shift(-horizon) / df["Close"] - 1
        df["Future_Return"] = future_return
        
        quantiles = config_target.get("quantiles", 5)
        def rank_quantiles(s):
            if s.notna().sum() < quantiles: return pd.Series(np.nan, index=s.index)
            return pd.qcut(s, q=quantiles, labels=False, duplicates='drop')
            
        df["Rank"] = df.groupby("Date")["Future_Return"].transform(rank_quantiles)
        df["target"] = np.nan
        df.loc[df["Rank"] == (quantiles - 1), "target"] = 1.0 
        df.loc[df["Rank"] == 0, "target"] = 0.0               
        
        df.loc[df["Future_Return"].isna(), "target"] = np.nan
        cols_to_drop = ["Future_Return", "Rank"]

    elif strategy == "trend_scanning":
        fwd_returns = pd.concat([(df.groupby("ticker")["Close"].shift(-h) / df["Close"]) - 1 for h in range(1, horizon + 1)], axis=1)
        time_arr = np.arange(1, horizon + 1)
        time_dev = time_arr - time_arr.mean()
        time_var = np.sum(time_dev**2)
        
        ret_mean = fwd_returns.mean(axis=1)
        ret_dev = fwd_returns.sub(ret_mean, axis=0)
        
        cov = ret_dev.dot(time_dev)
        slope = cov / time_var
        
        df["target"] = (slope > 0).astype(float)
        df.loc[fwd_returns.iloc[:, -1].isna(), "target"] = np.nan
        cols_to_drop = []

    elif strategy == "squeeze_breakout":
        if "Bollinger_Width_21d" not in df.columns:
            raise ValueError("Bollinger_Width_21d não encontrada.")
            
        future_return = df.groupby("ticker")["Close"].shift(-horizon) / df["Close"] - 1
        thresh_bb = df.groupby("ticker")["Bollinger_Width_21d"].transform(lambda x: x.quantile(0.20))
        cond_squeeze = df["Bollinger_Width_21d"] <= thresh_bb
        
        dynamic_thresh = np.maximum(threshold, vol_h * 0.5) if has_vol else threshold
        cond_breakout = future_return > dynamic_thresh
        
        df["target"] = (cond_squeeze & cond_breakout).astype(float)
        df.loc[future_return.isna(), "target"] = np.nan
        cols_to_drop = []

    elif strategy == "meta_labeling":
        primary_signal_col = config_target.get("primary_signal_col", "MACD_Hist")
        primary_signal_cond = config_target.get("primary_signal_cond", ">")
        primary_signal_val = config_target.get("primary_signal_val", 0)
        
        if primary_signal_col not in df.columns:
             if "RSI_14d" in df.columns:
                 primary_signal_col, primary_signal_cond, primary_signal_val = "RSI_14d", "<", 30
             else:
                 raise ValueError("Coluna de sinal primario nao encontrada para Meta-Labeling.")
        
        if primary_signal_cond == ">": signal = df[primary_signal_col] > primary_signal_val
        elif primary_signal_cond == "<": signal = df[primary_signal_col] < primary_signal_val
        else: signal = df[primary_signal_col] == primary_signal_val
            
        future_return = df.groupby("ticker")["Close"].shift(-horizon) / df["Close"] - 1
        profit_threshold = config_target.get("profit_threshold", 0.0)
        
        target = np.full(len(df), np.nan)
        mask_signal_true = signal.values
        ret_signal_true = future_return.values[mask_signal_true]
        
        win_mask = ret_signal_true > profit_threshold
        target[mask_signal_true] = np.where(win_mask, 1.0, 0.0)
        
        df["target"] = target
        df.loc[future_return.isna(), "target"] = np.nan
        cols_to_drop = []

    else:
        raise ValueError(f"Estratégia de target desconhecida: {strategy}")

    df = df.dropna(subset=["target"]).reset_index(drop=True)
    df["target"] = df["target"].astype(int)
    return df, cols_to_drop
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
        df_val = df[
            (df["Date_dt"] > data_train_end) & (df["Date_dt"] <= data_val_purged)
        ].copy()
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
        df_val = df[
            (df["Date_dt"] > data_train_end) & (df["Date_dt"] <= data_val_purged)
        ].copy()
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
    colunas_financas = [
        "Ativo_Total",
        "Caixa_Equivalentes",
        "Emprestimos_Curto_Prazo",
        "Emprestimos_Longo_Prazo",
        "Patrimonio_Liquido",
        "EBIT_Operacional",
        "Lucro_Prejuizo_Periodo",
        "Divida_Total",
    ]
    colunas_estaticos = ["beta", "marketCap", "trailingPE"]

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
                "balanco_yf": imp_dict,
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
                train_imp.loc[(train_imp["sector"] == s) & (train_imp[c].isna()), c] = (
                    medians.loc[s, c]
                )
                val_imp.loc[(val_imp["sector"] == s) & (val_imp[c].isna()), c] = (
                    medians.loc[s, c]
                )
                test_imp.loc[(test_imp["sector"] == s) & (test_imp[c].isna()), c] = (
                    medians.loc[s, c]
                )
        global_medians = train_imp[cols_f].median()
    elif strat_financas == "ticker_median":
        medians = train_imp.groupby("ticker")[cols_f].median()
        for s in medians.index:
            for c in cols_f:
                train_imp.loc[(train_imp["ticker"] == s) & (train_imp[c].isna()), c] = (
                    medians.loc[s, c]
                )
                val_imp.loc[(val_imp["ticker"] == s) & (val_imp[c].isna()), c] = (
                    medians.loc[s, c]
                )
                test_imp.loc[(test_imp["ticker"] == s) & (test_imp[c].isna()), c] = (
                    medians.loc[s, c]
                )
        global_medians = train_imp[cols_f].median()
        train_imp[cols_f] = train_imp[cols_f].fillna(global_medians).fillna(0)
        val_imp[cols_f] = val_imp[cols_f].fillna(global_medians).fillna(0)
        test_imp[cols_f] = test_imp[cols_f].fillna(global_medians).fillna(0)
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
        train_imp[cols_e] = train_imp[cols_e].fillna(global_medians).fillna(0)
        val_imp[cols_e] = val_imp[cols_e].fillna(global_medians).fillna(0)
        test_imp[cols_e] = test_imp[cols_e].fillna(global_medians).fillna(0)
    elif strat_estaticos == "sector_median":
        medians = train_imp.groupby("sector")[cols_e].median()
        for s in medians.index:
            for c in cols_e:
                train_imp.loc[(train_imp["sector"] == s) & (train_imp[c].isna()), c] = (
                    medians.loc[s, c]
                )
                val_imp.loc[(val_imp["sector"] == s) & (val_imp[c].isna()), c] = (
                    medians.loc[s, c]
                )
                test_imp.loc[(test_imp["sector"] == s) & (test_imp[c].isna()), c] = (
                    medians.loc[s, c]
                )
        global_medians = train_imp[cols_e].median()
        train_imp[cols_e] = train_imp[cols_e].fillna(global_medians).fillna(0)
        val_imp[cols_e] = val_imp[cols_e].fillna(global_medians).fillna(0)
        test_imp[cols_e] = test_imp[cols_e].fillna(global_medians).fillna(0)
    elif strat_estaticos == "ticker_median":
        medians = train_imp.groupby("ticker")[cols_e].median()
        for s in medians.index:
            for c in cols_e:
                train_imp.loc[(train_imp["ticker"] == s) & (train_imp[c].isna()), c] = (
                    medians.loc[s, c]
                )
                val_imp.loc[(val_imp["ticker"] == s) & (val_imp[c].isna()), c] = (
                    medians.loc[s, c]
                )
                test_imp.loc[(test_imp["ticker"] == s) & (test_imp[c].isna()), c] = (
                    medians.loc[s, c]
                )
        global_medians = train_imp[cols_e].median()
        train_imp[cols_e] = train_imp[cols_e].fillna(global_medians).fillna(0)
        val_imp[cols_e] = val_imp[cols_e].fillna(global_medians).fillna(0)
        test_imp[cols_e] = test_imp[cols_e].fillna(global_medians).fillna(0)
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
                train_imp.loc[(train_imp["sector"] == s) & (train_imp[c].isna()), c] = (
                    medians.loc[s, c]
                )
                val_imp.loc[(val_imp["sector"] == s) & (val_imp[c].isna()), c] = (
                    medians.loc[s, c]
                )
                test_imp.loc[(test_imp["sector"] == s) & (test_imp[c].isna()), c] = (
                    medians.loc[s, c]
                )
        global_medians = train_imp[cols_yf].median()
        train_imp[cols_yf] = train_imp[cols_yf].fillna(global_medians).fillna(0)
        val_imp[cols_yf] = val_imp[cols_yf].fillna(global_medians).fillna(0)
        test_imp[cols_yf] = test_imp[cols_yf].fillna(global_medians).fillna(0)
    elif strat_yf == "ticker_median":
        medians = train_imp.groupby("ticker")[cols_yf].median()
        for s in medians.index:
            for c in cols_yf:
                train_imp.loc[(train_imp["ticker"] == s) & (train_imp[c].isna()), c] = (
                    medians.loc[s, c]
                )
                val_imp.loc[(val_imp["ticker"] == s) & (val_imp[c].isna()), c] = (
                    medians.loc[s, c]
                )
                test_imp.loc[(test_imp["ticker"] == s) & (test_imp[c].isna()), c] = (
                    medians.loc[s, c]
                )
        global_medians = train_imp[cols_yf].median()
        train_imp[cols_yf] = train_imp[cols_yf].fillna(global_medians).fillna(0)
        val_imp[cols_yf] = val_imp[cols_yf].fillna(global_medians).fillna(0)
        test_imp[cols_yf] = test_imp[cols_yf].fillna(global_medians).fillna(0)
    else:
        # Fallback de Segurança Anti-Crash
        train_imp[cols_yf] = train_imp[cols_yf].fillna(0)
        val_imp[cols_yf] = val_imp[cols_yf].fillna(0)
        test_imp[cols_yf] = test_imp[cols_yf].fillna(0)

    return train_imp, val_imp, test_imp


def preprocessar_dados(
    df_raw,
    target_strategy,
    target_definition,
    split_config,
    inf_handling_strategy,
    imputation_strategy,
    outlier_handling,
    scaling_method,
):
    """
    Passo 1: Faz a limpeza, split e escala da base apenas 1 vez.
    Retorna as matrizes prontas para a rede neural.
    """
    df_target, cols_to_drop = gerar_target(df_raw, target_strategy, target_definition)

    metadata_cols = [
        "Date",
        "ticker",
        "target",
        "sector",
        "industry",
        "tamanho_categoria",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Stock Splits",
        "Retorno",
        "Dividends",
        "Dividends Payable"
    ] + cols_to_drop
    cols_to_exclude = [col for col in metadata_cols if col in df_target.columns]
    feature_cols = [
        col
        for col in df_target.columns
        if col not in cols_to_exclude and pd.api.types.is_numeric_dtype(df_target[col])
    ]

    if inf_handling_strategy == "replace_nan":
        df_target[feature_cols] = df_target[feature_cols].replace(
            [np.inf, -np.inf], np.nan
        )
    elif inf_handling_strategy == "drop_row":
        df_target = df_target[
            ~np.isinf(df_target[feature_cols]).any(axis=1)
        ].reset_index(drop=True)

    horizon_days = target_definition.get("horizon_days", 10)
    df_train, df_val, df_test = split_temporal(
        df_target, split_config, horizon_days=horizon_days
    )

    if isinstance(imputation_strategy, str):
        imputation_strategy = {
            "financas": imputation_strategy,
            "estaticos": imputation_strategy,
            "balanco_yf": imputation_strategy,
        }

    df_train, df_val, df_test = imputar_nulos_granular(
        df_train, df_val, df_test, feature_cols, imputation_strategy
    )

    if outlier_handling == "clip_99_1":
        lower = df_train[feature_cols].quantile(0.01)
        upper = df_train[feature_cols].quantile(0.99)
        df_train[feature_cols] = df_train[feature_cols].clip(
            lower=lower, upper=upper, axis=1
        )
        df_val[feature_cols] = df_val[feature_cols].clip(
            lower=lower, upper=upper, axis=1
        )
        df_test[feature_cols] = df_test[feature_cols].clip(
            lower=lower, upper=upper, axis=1
        )

    X_train, y_train = df_train[feature_cols].copy(), df_train["target"]
    X_val, y_val = df_val[feature_cols].copy(), df_val["target"]
    X_test, y_test = df_test[feature_cols].copy(), df_test["target"]

    if scaling_method != "none" and len(feature_cols) > 0:
        if scaling_method == "standard":
            scaler = StandardScaler()
        elif scaling_method == "minmax":
            scaler = MinMaxScaler()
        elif scaling_method == "robust":
            scaler = RobustScaler()
        X_train[feature_cols] = scaler.fit_transform(X_train[feature_cols])
        X_val[feature_cols] = scaler.transform(X_val[feature_cols])
        X_test[feature_cols] = scaler.transform(X_test[feature_cols])

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols


def treinar_e_avaliar_modelo_pre_processado(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    feature_cols,
    model_name,
    hparams,
    calculate_permutation_importance=True,
    fs_cache=None,
    force_cpu=False,
):
    """
    Passo 2: Pega os dados 100% limpos, faz o Feature Selection e treina na GPU/CPU.
    """
    # 5. Feature Selection
    drop_feat = hparams.get("drop_feat", None)
    
    clean_hparams = {
        k: v
        for k, v in hparams.items()
        if not k.startswith("_") and k != "drop_feat"
    }

    # Helper para device override
    local_device = "cpu" if force_cpu else DEVICE
    
    k_cut = None
    is_pca = False
    n_comp = 1
    
    if drop_feat is not None and isinstance(drop_feat, dict):
        df_type = drop_feat.get("type", "none")
        if df_type != "none":
            k_cut = drop_feat.get("k", None)
            if df_type == "pca":
                is_pca = True
                n_comp = drop_feat.get("n_components", 1)

    if k_cut is not None and k_cut < len(feature_cols):
        cache_key = f"drop_feat_{'pca' if is_pca else 'top_k'}_{k_cut}_{n_comp}"
        
        with fs_global_lock:
            cached_result = fs_cache.get(cache_key) if fs_cache is not None else None
            
        if cached_result is not None:
            feature_cols = cached_result['cols']
            X_train = cached_result['X_train']
            X_val = cached_result['X_val']
            X_test = cached_result['X_test']
        else:
            selector = SelectFromModel(
                DecisionTreeClassifier(random_state=42, max_depth=10),
                max_features=k_cut,
                threshold=-np.inf,
            )
            selector.fit(X_train[feature_cols], y_train)
            support = selector.get_support()
            
            top_cols = np.array(feature_cols)[support].tolist()
            rejected_cols = np.array(feature_cols)[~support].tolist()
            
            if is_pca and len(rejected_cols) > n_comp:
                pca = PCA(n_components=n_comp, random_state=42)
                pca.fit(X_train[rejected_cols])
                
                def apply_pca(df, pca_model, rej_cols, n):
                    pca_df = pd.DataFrame(
                        pca_model.transform(df[rej_cols]),
                        index=df.index,
                        columns=[f"pca_residual_{i+1}" for i in range(n)]
                    )
                    return pd.concat([df[top_cols], pca_df], axis=1)
                
                X_train = apply_pca(X_train, pca, rejected_cols, n_comp)
                X_val = apply_pca(X_val, pca, rejected_cols, n_comp)
                X_test = apply_pca(X_test, pca, rejected_cols, n_comp)
                
                feature_cols = top_cols + [f"pca_residual_{i+1}" for i in range(n_comp)]
            else:
                feature_cols = top_cols
                X_train, X_val, X_test = X_train[feature_cols], X_val[feature_cols], X_test[feature_cols]

            if fs_cache is not None:
                with fs_global_lock:
                    fs_cache[cache_key] = {
                        'cols': feature_cols,
                        'X_train': X_train,
                        'X_val': X_val,
                        'X_test': X_test
                    }

    # 6. Instanciar Modelo (Reprodutibilidade: Random State 42 sempre que possível)
    if "random_state" not in clean_hparams and model_name not in [
        "knn",
        "voting_classifier",
    ]:
        clean_hparams["random_state"] = 42

    # Estratégia A: Força n_jobs=1 sempre para evitar Thread Oversubscription
    if model_name in ["xgboost", "lightgbm", "random_forest", "voting_classifier"]:
        clean_hparams["n_jobs"] = 1

    if model_name == "decision_tree":
        model = DecisionTreeClassifier(**clean_hparams)
    elif model_name == "random_forest":
        model = RandomForestClassifier(**clean_hparams)
    elif model_name == "knn":
        model = PyTorchKNNWrapper(device=local_device, **clean_hparams)
    elif model_name == "mlp":
        model = SklearnPyTorchMLPWrapper(device=local_device, **clean_hparams)
    elif model_name == "logistic_regression":
        model = SklearnPyTorchLogisticRegressionWrapper(device=local_device, **clean_hparams)
    elif model_name == "lightgbm":
        if LIGHTGBM_AVAILABLE:
            if torch.cuda.is_available() and local_device != "cpu":
                clean_hparams["device"] = "gpu"
            if "verbose" not in clean_hparams:
                clean_hparams["verbose"] = -1
            clean_hparams["force_row_wise"] = True
            
            # Correção vital do LightGBM: ele ignora o 'subsample' se o 'subsample_freq' for 0 (padrão)
            if "subsample" in clean_hparams and clean_hparams["subsample"] < 1.0:
                if "subsample_freq" not in clean_hparams or clean_hparams["subsample_freq"] == 0:
                    clean_hparams["subsample_freq"] = 1
                    
            model = LGBMClassifier(**clean_hparams)
        else:
            raise ValueError("LightGBM não está instalado.")
    elif model_name == "xgboost":
        # Ativação nativa da GPU para o XGBoost
        if torch.cuda.is_available() and local_device != "cpu":
            clean_hparams["tree_method"] = "hist"
            clean_hparams["device"] = "cuda"
        if "verbosity" not in clean_hparams:
            clean_hparams["verbosity"] = 0
        if XGBOOST_AVAILABLE:
            model = XGBClassifier(**clean_hparams)
        else:
            model = RandomForestClassifier(**clean_hparams)
    elif model_name == "voting_classifier":
        # Habilitar GPU para os componentes do Voting
        xgb_params = {
            "n_estimators": 100,
            "max_depth": 5,
            "learning_rate": 0.05,
            "random_state": 42,
            "n_jobs": 1,
        }
        if torch.cuda.is_available() and local_device != "cpu":
            xgb_params["tree_method"] = "hist"
            xgb_params["device"] = "cuda"

        estimators = [
            (
                "xgb",
                XGBClassifier(**xgb_params)
                if XGBOOST_AVAILABLE
                else RandomForestClassifier(random_state=42, n_jobs=1),
            ),
            (
                "mlp",
                SklearnPyTorchMLPWrapper(
                    hidden_layer_sizes=(100,), max_iter=1000, random_state=42, device=local_device
                ),
            ),
            ("knn", PyTorchKNNWrapper(n_neighbors=5, device=local_device)),
            (
                "lg",
                SklearnPyTorchLogisticRegressionWrapper(max_iter=1000, random_state=42, device=local_device),
            ),
        ]
        voting_weights = clean_hparams.pop("weights", None)
        model = GPUVotingClassifier(estimators=estimators, weights=voting_weights)
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
        "test_f1_macro": f1_score(
            y_test, y_test_pred, average="macro", zero_division=0
        ),
        "test_f1_weighted": f1_score(
            y_test, y_test_pred, average="weighted", zero_division=0
        ),
        "test_precision_macro": precision_score(
            y_test, y_test_pred, average="macro", zero_division=0
        ),
        "test_recall_macro": recall_score(
            y_test, y_test_pred, average="macro", zero_division=0
        ),
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
        for idx in indices[
            :20
        ]:  # Salva as top 20 para economizar banco, ou podemos salvar tudo.
            if importances[idx] > 0:
                feature_importances[feature_cols[idx]] = float(importances[idx])

    # Tentativa 2: Modelos Lineares (Coeficientes)
    elif hasattr(model, "coef_"):
        # Pegamos a magnitude absoluta dos coeficientes (para classificação multi-classe, tiramos a média absoluta)
        coefs = (
            np.mean(np.abs(model.coef_), axis=0)
            if len(model.coef_.shape) > 1
            else np.abs(model.coef_[0])
        )
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
                result = permutation_importance(
                    model, X_val, y_val, n_repeats=5, random_state=42, n_jobs=1
                )
                importances = result.importances_mean
                indices = np.argsort(importances)[::-1]
                for idx in indices[:20]:
                    if importances[idx] > 0:
                        feature_importances[feature_cols[idx]] = float(importances[idx])
        else:
            importance_type = "skipped"

    if hasattr(model, "n_iter_"):
        feature_importances["_n_iter_executadas"] = float(model.n_iter_)

    return metrics, confusion_mat, feature_importances, importance_type
