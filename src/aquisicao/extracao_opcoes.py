import os
import json
import yaml
import time
from datetime import datetime
import pandas as pd
import yfinance as yf

# Delay padrão para evitar bloqueios na API do Yahoo Finance
DOWNLOAD_DELAY = 1


def extracao_opcoes():
    """
    Extrai as cadeias de opções (Calls e Puts) disponíveis hoje para os
    ativos configurados. Faz carga incremental diária (Delta Load).
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    # --- Dinâmica de Aquisição ---
    config_file = os.path.join(os.path.dirname(__file__), "companhias.yaml")

    # Se foi chamado pelo orquestrador, usa a pasta que o orquestrador mandou
    if "AQUISICAO_TARGET_DIR" in os.environ:
        output_dir = os.environ["AQUISICAO_TARGET_DIR"]
    else:
        # Se foi rodado manualmente (F5), salva solto no raw
        output_dir = os.path.join(project_root, "data", "raw")

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "opcoes.parquet")

    # 1. Leitura das companhias
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            tickers = [c.get("ticker") for c in data["companies"] if c.get("ticker")]
    except Exception as e:
        print(f"Erro ao ler companhias.yaml: {e}")
        return

    data_extracao = datetime.now().date()
    print(
        f"Iniciando extração de opções para a data {data_extracao} ({len(tickers)} ativos)..."
    )

    lista_opcoes = []

    for idx, ticker in enumerate(tickers):
        print(f"[{idx + 1}/{len(tickers)}] Buscando opções de {ticker}...")
        time.sleep(DOWNLOAD_DELAY)

        try:
            acao = yf.Ticker(ticker)
            expirations = acao.options

            if not expirations:
                print(f"Nenhuma opção disponível para {ticker}.")
                continue

            for exp in expirations:
                # Opcional: sleep pequeno entre vencimentos da mesma ação para não bloquear
                time.sleep(0.5)

                chain = acao.option_chain(exp)

                # Trata Calls (Opções de Compra)
                if not chain.calls.empty:
                    df_calls = chain.calls.copy()
                    df_calls["Tipo"] = "Call"
                    df_calls["Vencimento"] = exp
                    df_calls["ticker"] = ticker
                    df_calls["Data_Extracao"] = data_extracao
                    lista_opcoes.append(df_calls)

                # Trata Puts (Opções de Venda)
                if not chain.puts.empty:
                    df_puts = chain.puts.copy()
                    df_puts["Tipo"] = "Put"
                    df_puts["Vencimento"] = exp
                    df_puts["ticker"] = ticker
                    df_puts["Data_Extracao"] = data_extracao
                    lista_opcoes.append(df_puts)

        except Exception as e:
            print(f"Erro ao processar opções de {ticker}: {e}")

    # 2. Consolidação e Salvamento
    if lista_opcoes:
        df_new = pd.concat(lista_opcoes, ignore_index=True)
        # Convertendo as colunas de data/hora que vieram do yfinance para datetime sem timezone
        if "lastTradeDate" in df_new.columns:
            df_new["lastTradeDate"] = pd.to_datetime(
                df_new["lastTradeDate"]
            ).dt.tz_localize(None)

        df_new["Data_Extracao"] = pd.to_datetime(df_new["Data_Extracao"])

        # --- Carga Incremental (Delta Load) ---
        if os.path.exists(output_file):
            print("[Delta Load] Arquivo de opções encontrado. Anexando dados do dia...")
            df_old = pd.read_parquet(output_file)

            # Para evitar duplicatas caso o script rode duas vezes no mesmo dia
            # Filtra todas as datas antigas ignorando a extração do dia atual, e junta com os dados recém-baixados
            df_old = df_old[df_old["Data_Extracao"] != pd.to_datetime(data_extracao)]

            df_final = pd.concat([df_old, df_new], ignore_index=True)
        else:
            print("[Full Load] Criando histórico de opções pela primeira vez...")
            df_final = df_new

        df_final.to_parquet(output_file, index=False)
        print(f"Carga de opções concluída! Salvo em: {output_file}")
    else:
        print("Nenhum dado de opções foi retornado para a lista de ativos.")


if __name__ == "__main__":
    extracao_opcoes()
