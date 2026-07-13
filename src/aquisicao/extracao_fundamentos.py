import os
import glob
import json
import time
import pandas as pd
import yfinance as yf

# Delay padrão entre requisições para não sobrecarregar a API
DOWNLOAD_DELAY = 1

def extracao_fundamentos():
    """
    Extrai dados fundamentalistas (Balanço, DRE, Fluxo de Caixa e Info) 
    do Yahoo Finance para os ativos configurados em companhias.json.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    # --- Dinâmica de Aquisição ---
    config_file = os.path.join(project_root, "config", "companhias.json")
    
    # Se foi chamado pelo orquestrador, usa a pasta que o orquestrador mandou
    if "AQUISICAO_TARGET_DIR" in os.environ:
        output_dir = os.environ["AQUISICAO_TARGET_DIR"]
    else:
        # Se foi rodado manualmente (F5), salva solto no raw
        output_dir = os.path.join(project_root, "data", "raw")
    
    os.makedirs(output_dir, exist_ok=True)
    
    info_output = os.path.join(output_dir, "fundamentos_info.parquet")
    balanco_output = os.path.join(output_dir, "fundamentos_balanco.parquet")

    # 1. Leitura das companhias
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            tickers = [c.get("ticker") for c in data["companies"] if c.get("ticker")]
    except Exception as e:
        print(f"Erro ao ler companhias.json: {e}")
        return

    print(f"Iniciando extração de fundamentos para {len(tickers)} ativos...")

    lista_info = []
    lista_balancos = []

    for idx, ticker in enumerate(tickers):
        print(f"[{idx + 1}/{len(tickers)}] Baixando fundamentos de {ticker}...")
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
            
            # --- Extração do Balanço Patrimonial (Balance Sheet) ---
            # Para extrair outros como financials e cashflow, podemos usar acao.financials, acao.cashflow
            df_bs = acao.balance_sheet
            if not df_bs.empty:
                # Transpõe para que a data vire coluna e reseta o índice (nome da métrica contábil)
                df_bs = df_bs.T.reset_index().rename(columns={'index': 'Data_Referencia'})
                # Derrete (melt) o dataframe para o formato longo (Tidy)
                df_bs_long = df_bs.melt(id_vars=['Data_Referencia'], var_name='Metrica', value_name='Valor')
                df_bs_long['ticker'] = ticker
                lista_balancos.append(df_bs_long)
                
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

if __name__ == "__main__":
    extracao_fundamentos()
