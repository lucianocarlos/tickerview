import os
import glob
import json
import yaml
import time
import pandas as pd
import yfinance as yf

# Delay padrão entre requisições para não sobrecarregar a API
DOWNLOAD_DELAY = 1

def extracao_fundamentos():
    """
    Extrai dados fundamentalistas (Balanço, DRE, Fluxo de Caixa e Info) 
    do Yahoo Finance para os ativos configurados em companhias.yaml.
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
    
    info_output = os.path.join(output_dir, "fundamentos_info.parquet")
    balanco_output = os.path.join(output_dir, "fundamentos_balanco.parquet")
    metadados_output = os.path.join(output_dir, "metadados.parquet")

    # 1. Leitura das companhias
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            tickers = [c.get("ticker") for c in data["companies"] if c.get("ticker")]
    except Exception as e:
        print(f"Erro ao ler companhias.yaml: {e}")
        return

    print(f"Iniciando extração de fundamentos para {len(tickers)} ativos...")

    lista_info = []
    lista_balancos = []
    lista_metadados = []

    for idx, ticker in enumerate(tickers):
        print(f"[{idx + 1}/{len(tickers)}] Baixando fundamentos e metadados de {ticker}...")
        time.sleep(DOWNLOAD_DELAY)
        
        try:
            acao = yf.Ticker(ticker)
            
            # --- Extração do Info (Cross-sectional estático) ---
            info_data = acao.info
            if info_data:
                # Transforma o dicionário em formato Tidy (Chave-Valor)
                df_info = pd.DataFrame(list(info_data.items()), columns=['Metrica', 'Valor'])
                df_info['ticker'] = ticker
                df_info['Data_Extracao'] = pd.Timestamp.now().normalize()
                # Converte o valor para string para evitar conflito de tipos no Parquet (ex: "Infinity")
                df_info['Valor'] = df_info['Valor'].astype(str)
                lista_info.append(df_info)
                
                # --- Extração de Metadados (substituindo extracao_metadados.py) ---
                market_cap = info_data.get("marketCap")
                sector = info_data.get("sector")
                industry = info_data.get("industry")
                beta = info_data.get("beta")

                if market_cap is None:
                    tamanho_categoria = "Desconhecido"
                elif market_cap >= 10_000_000_000:
                    tamanho_categoria = "Large Cap"
                elif market_cap >= 2_000_000_000:
                    tamanho_categoria = "Mid Cap"
                else:
                    tamanho_categoria = "Small Cap"

                metadata = {
                    "ticker": ticker,
                    "sector": sector,
                    "industry": industry,
                    "marketCap": market_cap,
                    "beta": beta,
                    "tamanho_categoria": tamanho_categoria,
                }
                lista_metadados.append(metadata)
            
            # --- Extração do Balanço Patrimonial (Anual e Trimestral) ---
            df_bs = acao.balance_sheet
            df_qbs = acao.quarterly_balance_sheet
            
            if not df_bs.empty:
                df_bs = df_bs.T.reset_index().rename(columns={'index': 'Data_Referencia'})
                df_bs_long = df_bs.melt(id_vars=['Data_Referencia'], var_name='Metrica', value_name='Valor')
                df_bs_long['ticker'] = ticker
                df_bs_long['Tipo_Balanco'] = 'Anual'
                lista_balancos.append(df_bs_long)
                
            if not df_qbs.empty:
                df_qbs = df_qbs.T.reset_index().rename(columns={'index': 'Data_Referencia'})
                df_qbs_long = df_qbs.melt(id_vars=['Data_Referencia'], var_name='Metrica', value_name='Valor')
                df_qbs_long['ticker'] = ticker
                df_qbs_long['Tipo_Balanco'] = 'Trimestral'
                lista_balancos.append(df_qbs_long)
                
        except Exception as e:
            print(f"Erro ao processar {ticker}: {e}")

    # 2. Salva o Info em Parquet
    if lista_info:
        df_info_final = pd.concat(lista_info, ignore_index=True)
        df_info_final.to_parquet(info_output, index=False)
        print(f"Arquivo de info salvo: {info_output}")

    # 3. Salva os Balanços em Parquet (Formato Longo)
    if lista_balancos:
        df_balancos_final = pd.concat(lista_balancos, ignore_index=True)
        # Limpeza básica na data
        if pd.api.types.is_datetime64_any_dtype(df_balancos_final['Data_Referencia']):
             df_balancos_final["Data_Referencia"] = df_balancos_final["Data_Referencia"].dt.tz_localize(None)
        df_balancos_final.to_parquet(balanco_output, index=False)
        print(f"Arquivo de balanços salvo: {balanco_output}")

    # 4. Salva Metadados em Parquet
    if lista_metadados:
        df_metadados_final = pd.DataFrame(lista_metadados)
        df_metadados_final.to_parquet(metadados_output, index=False)
        print(f"Arquivo de metadados salvo: {metadados_output}")

if __name__ == "__main__":
    extracao_fundamentos()
