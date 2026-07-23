import os
import json
import yaml
import time
import pandas as pd
import yfinance as yf
from datetime import timedelta

# Data de início padrão para a extração histórica
START_DATE = "2010-01-01"

# Limite final da extração (dias de deslocamento em relação a hoje):
# 0 para Hoje, -1 para Ontem (D-1), -2 para Antes de Ontem, etc.
END_DATE = -1

# Delay padrão antes de iniciar o download (em milissegundos)
DOWNLOAD_DELAY = 2


def process_yfinance_data(df, ticker=None):
    """Transforma o formato largo do yfinance em formato longo (Tidy Data)."""
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df = (
            df.stack(level=1, future_stack=True)
            .rename_axis(["Date", "ticker"])
            .reset_index()
        )
    else:
        if "Date" not in df.columns:
            df = df.reset_index()
        if "ticker" not in df.columns and ticker is not None:
            df["ticker"] = ticker

    # Para evitar divergências de escala (pois YFinance ajusta o Close para dividendos no 'Adj Close' 
    # mas não ajusta Open/High/Low), nós vamos calcular o fator de ajuste e aplicar em todas.
    if "Adj Close" in df.columns and "Close" in df.columns:
        fator_ajuste = df["Adj Close"] / df["Close"].replace(0, 1)
        df["Open"] = df["Open"] * fator_ajuste
        df["High"] = df["High"] * fator_ajuste
        df["Low"] = df["Low"] * fator_ajuste
        # O Close oficial agora passa a ser o Adj Close perfeitamente ajustado
        df["Close"] = df["Adj Close"]

    col_mapping = {
        "Open": "Open",
        "High": "High",
        "Low": "Low",
        "Close": "Close",
        "Volume": "Volume",
    }
    df = df.rename(columns=col_mapping)
    cols_to_keep = [
        "Date",
        "ticker",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]
    return df[[c for c in cols_to_keep if c in df.columns]]


def download_prices_by_ticker(tickers, start, end):
    """
    Downloads prices ticker by ticker with the configured delay and retry logic.
    """
    dfs = []
    total = len(tickers)
    for idx, ticker in enumerate(tickers):
        print(f"\n[*] Progresso: {idx + 1}/{total} - Iniciando extração de {ticker}")

        df_ticker = pd.DataFrame()
        for attempt in range(1, 4):
            # Sempre espera o delay padrão (DOWNLOAD_DELAY) em segundos antes de tentar
            if DOWNLOAD_DELAY > 0:
                print(f"[*] Aguardando delay de {DOWNLOAD_DELAY} segundos...")
                time.sleep(DOWNLOAD_DELAY)

            try:
                print(f"[*] Fazendo download de {ticker} (Tentativa {attempt}/3)...")
                # keepna baixa as tuplas nulas do yahoo finance
                df_ticker = yf.download(
                    ticker, start=start, end=end, progress=False, keepna=True
                )

                # yfinance retorna df vazio ou sem linhas se falhar
                if not df_ticker.empty and not (
                    len(df_ticker) == 1 and df_ticker.isna().all().all()
                ):
                    break
                else:
                    print(
                        f"[!] Tentativa {attempt}/3 para {ticker}: Nenhum dado retornado ou inválido."
                    )
                    df_ticker = pd.DataFrame()
            except Exception as e:
                print(f"[!] Erro no download de {ticker} (Tentativa {attempt}/3): {e}")
                df_ticker = pd.DataFrame()

        if not df_ticker.empty:
            df_processed = process_yfinance_data(df_ticker, ticker)
            dfs.append(df_processed)
        else:
            print(
                f"[WARNING] Falha definitiva no download do ativo {ticker} após 3 tentativas."
            )

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


def fetch_prices():

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    # --- Dinâmica de Aquisição ---
    config_file = os.path.join(os.path.dirname(__file__), "companhias.yaml")

    # Se foi chamado pelo orquestrador, usa a pasta que o orquestrador mandou
    if "AQUISICAO_TARGET_DIR" in os.environ:
        output_dir = os.environ["AQUISICAO_TARGET_DIR"]
    else:
        # Se foi rodado manualmente (F5), salva solto no raw
        output_dir = os.path.join(project_root, "data", "raw")
    output_file = os.path.join(output_dir, "precos.parquet")

    os.makedirs(output_dir, exist_ok=True)

    # --- LEITURA DO JSON ---
    try:
        with open(config_file, "r") as f:
            data = yaml.safe_load(f)
            # vai montar uma lista de companies, se esta tiver um ticker definido no dataset
            # c.get("ticker") (o primeiro), extrai epenas o valor de 'tikcer', mantando assim uma lista de tickers
            tickers = [c.get("ticker") for c in data["companies"] if c.get("ticker")]
    except Exception as e:
        print(f"Erro ao ler companhias.yaml: {e}")
        return

    print(f"Iniciando extração em lote para {len(tickers)} ativos...")

    # --- O NOVO MOTOR DE CARGA INCREMENTAL (DELTA LOAD) ---
    if os.path.exists(output_file):
        print("[Delta Load] Arquivo Parquet encontrado. Lendo histórico...")
        df_old = pd.read_parquet(output_file)

        # Garante timezone limpo para evitar conflitos no Parquet
        df_old["Date"] = pd.to_datetime(df_old["Date"]).dt.tz_localize(None)

        # O início será o maior entre a última data do banco ou a data configurada (START_DATE)
        last_date = max(
            df_old["Date"].max(), pd.Timestamp(START_DATE) - timedelta(days=1)
        )

        # Determina a data limite final baseada em END_DATE (deslocamento em dias)
        end_limit = pd.Timestamp.now().normalize() + timedelta(days=int(END_DATE))

        if last_date >= end_limit:
            print(f"Última data no banco: {df_old['Date'].max().strftime('%Y-%m-%d')}")
            print(
                f"Data limite final configurada ({END_DATE}): {end_limit.strftime('%Y-%m-%d')}"
            )
            print(
                "Nenhum pregão novo disponível para baixar (banco de dados já atualizado)."
            )
            return

        start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
        end_query = (end_limit + timedelta(days=1)).strftime("%Y-%m-%d")

        print(f"Última data no banco: {df_old['Date'].max().strftime('%Y-%m-%d')}")
        print(
            f"Baixando dados no intervalo: {start_date} até {end_limit.strftime('%Y-%m-%d')}"
        )

        # Download por ticker com retry e delay
        df_new_raw = download_prices_by_ticker(tickers, start=start_date, end=end_query)

        if df_new_raw.empty or (len(df_new_raw) == 1 and df_new_raw.isna().all().all()):
            print("Nenhum pregão novo encontrado. Banco já está atualizado.")
            return

        df_new = process_yfinance_data(df_new_raw)
        df_new["Date"] = pd.to_datetime(df_new["Date"]).dt.tz_localize(None)

        df_final = pd.concat([df_old, df_new]).drop_duplicates(
            subset=["Date", "ticker"], keep="last"
        )
        print("Adicionadas novas linhas. Salvando Parquet...")
        df_final.to_parquet(output_file, index=False)

    else:  # não existe o dataset parquet
        # Determina a data limite final baseada em END_DATE (deslocamento em dias)
        end_limit = pd.Timestamp.now().normalize() + timedelta(days=int(END_DATE))

        print(
            f"[Full Load] Parquet não encontrado. Baixando histórico de {START_DATE} até {end_limit.strftime('%Y-%m-%d')}..."
        )
        end_query = (end_limit + timedelta(days=1)).strftime("%Y-%m-%d")
        df_new_raw = download_prices_by_ticker(tickers, start=START_DATE, end=end_query)
        df_final = process_yfinance_data(df_new_raw)
        df_final["Date"] = pd.to_datetime(df_final["Date"]).dt.tz_localize(None)

        df_final.to_parquet(output_file, index=False)
        print("Carga histórica completa finalizada com sucesso!")


if __name__ == "__main__":
    fetch_prices()
