import os
import json
import yaml
import pandas as pd
import yfinance as yf


def fetch_metadata():
    print("Iniciando Extração de Metadados...")

    # Estruturação de caminhos absolutos
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    # --- Dinâmica de Aquisição ---
    config_file = os.path.join(os.path.dirname(__file__), "companhias.yaml")

    # Se foi chamado pelo orquestrador, usa a pasta que o orquestrador mandou
    if "AQUISICAO_TARGET_DIR" in os.environ:
        output_dir = os.environ["AQUISICAO_TARGET_DIR"]
    else:
        # Se foi rodado manualmente (F5), salva solto no raw
        output_dir = os.path.join(project_root, "data", "raw")
    output_file = os.path.join(output_dir, "metadados.parquet")

    # Garantir que a pasta de destino exista
    os.makedirs(output_dir, exist_ok=True)

    # Lendo o companhias.yaml para extrair os tickers
    print(f"Lendo base de ativos em {config_file}...")
    try:
        with open(config_file, "r") as f:
            data = yaml.safe_load(f)

        tickers = []
        if isinstance(data, dict) and "companies" in data:
            tickers = [c.get("ticker") for c in data["companies"] if c.get("ticker")]
        else:
            print("Erro: A estrutura do companhias.yaml não possui a lista 'companies'.")
            return

        if not tickers:
            print("Aviso: Nenhum ticker encontrado no arquivo de configuração.")
            return

    except Exception as e:
        print(f"Erro ao ler companhias.yaml: {e}")
        return

    # Lista vazia para armazenar os dados de metadados
    all_metadata = []

    # Loop pelos tickers do universe
    print(f"Iniciando download para {len(tickers)} ativos...")
    for ticker in tickers:
        print(f"Extraindo metadados de {ticker}...")
        try:
            # Obtendo informações fundamentais da biblioteca yfinance
            info_dict = yf.Ticker(ticker).info

            # Extraindo dados de forma segura com .get()
            market_cap = info_dict.get("marketCap")
            sector = info_dict.get("sector")
            industry = info_dict.get("industry")
            beta = info_dict.get("beta")

            # Lógica para categorização do Tamanho de Capitalização
            if market_cap is None:
                tamanho_categoria = "Desconhecido"
            elif market_cap >= 10_000_000_000:
                tamanho_categoria = "Large Cap"
            elif market_cap >= 2_000_000_000:
                tamanho_categoria = "Mid Cap"
            else:
                tamanho_categoria = "Small Cap"

            # Inserindo metadados em um dicionário de resultados
            metadata = {
                "ticker": ticker,
                "sector": sector,
                "industry": industry,
                "marketCap": market_cap,
                "beta": beta,
                "tamanho_categoria": tamanho_categoria,
            }

            all_metadata.append(metadata)

        except Exception as e:
            print(f"Aviso: Erro ao extrair metadados para o ativo {ticker}: {e}")
            continue

    # Agrupando em um DataFrame Pandas
    if not all_metadata:
        print("Erro: Nenhum dado de metadados foi acumulado durante a execução.")
        return

    print("\nConsolidando tabela final de metadados...")
    df_metadata = pd.DataFrame(all_metadata)

    # Salvando em formato Parquet
    print(f"Salvando dados estruturados em {output_file}...")
    df_metadata.to_parquet(output_file, index=False)

    print("Processo concluído com sucesso!")


if __name__ == "__main__":
    fetch_metadata()
