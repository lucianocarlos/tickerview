import os
import glob
import json
import time
import requests
import pandas as pd
from datetime import date

BRAPI_TOKEN = 'COLOQUE_SEU_TOKEN_AQUI'

def fetch_fundamentals():
    # Define os caminhos
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    # --- Dinâmica de Aquisição ---
    raw_base = os.path.join(project_root, "data", "raw")
    aquisicoes = glob.glob(os.path.join(raw_base, "aquisicao_*"))
    aquisicao_atual = os.path.basename(sorted(aquisicoes)[-1]) if aquisicoes else "aquisicao_001"
    aquisicao_dir = os.path.join(raw_base, aquisicao_atual)
    os.makedirs(aquisicao_dir, exist_ok=True)
    config_file = os.path.join(project_root, 'config', 'companhias.json')
    output_dir = os.path.join(aquisicao_dir, 'fundamentals')
    output_file = os.path.join(output_dir, 'fundamentals.parquet')
    
    # Cria diretório de saída caso não exista
    os.makedirs(output_dir, exist_ok=True)
    
    # Lê os ativos
    try:
        with open(config_file, 'r') as f:
            data = json.load(f)
            
        # Extrai a lista de tickers do novo formato 'companies', mantendo fallback
        tickers = []
        if isinstance(data, dict) and 'companies' in data:
            tickers = [c.get('ticker') for c in data['companies'] if c.get('ticker')]
        elif isinstance(data, list):
            tickers = data
        elif isinstance(data, dict) and 'tickers' in data:
            tickers = data['tickers']
        else:
            print("Erro: Formato não suportado em universe.json.")
            return
            
    except FileNotFoundError:
        print(f"Erro: Arquivo {config_file} não encontrado.")
        return
    except json.JSONDecodeError:
        print(f"Erro: Arquivo {config_file} possui um formato JSON inválido.")
        return
        
    all_data = []
    
    for ticker in tickers:
        print(f"Buscando fundamentos para {ticker}...")
        
        try:
            url = f"https://brapi.dev/api/quote/{ticker}?fundamental=true&token={BRAPI_TOKEN}"
            response = requests.get(url, timeout=10)
            
            # Lança uma exceção se a requisição não foi bem sucedida (ex: 404, 500)
            response.raise_for_status()
            
            json_data = response.json()
            
            if 'results' not in json_data or not json_data['results']:
                print(f"Aviso: Resultados não encontrados para {ticker} na resposta da API.")
                continue
                
            result = json_data['results'][0]
            
            # Extração dos dados fundamentalistas
            # Busca as métricas principais do retorno da API. 
            # Dependendo da estrutura da Brapi, esses dados podem estar na raiz ou dentro de algum objeto aninhado.
            fundamental_data = {
                'ticker': ticker,
                'date_report': date.today(),
                'priceEarnings': result.get('priceEarnings'), # P/L
                'priceToBook': result.get('priceToBook'),     # VPA
                'returnOnEquity': result.get('returnOnEquity'), # ROE
                'dividendYield': result.get('dividendYield'),
                'debtToEquity': result.get('debtToEquity'),   # Dívida/Patrimônio
                # Adicionando outros campos úteis se existirem
                'regularMarketPrice': result.get('regularMarketPrice'),
                'marketCap': result.get('marketCap')
            }
            
            all_data.append(fundamental_data)
            
        except requests.exceptions.HTTPError as errh:
            print(f"Erro HTTP para {ticker}: {errh}")
        except requests.exceptions.ConnectionError as errc:
            print(f"Erro de Conexão para {ticker}: {errc}")
        except requests.exceptions.Timeout as errt:
            print(f"Timeout para {ticker}: {errt}")
        except Exception as e:
            print(f"Erro inesperado ao buscar {ticker}: {e}")
            
        # Pausa de 1 segundo entre as requisições para respeitar o rate limit da Brapi no plano gratuito
        time.sleep(1)
        
    if not all_data:
        print("Nenhum dado de fundamentos foi baixado.")
        return
        
    print("Combinando os dados...")
    # Cria o DataFrame a partir da lista de dicionários
    final_df = pd.DataFrame(all_data)
    
    # Converte 'date_report' para datetime antes de salvar como parquet para garantir o tipo correto
    final_df['date_report'] = pd.to_datetime(final_df['date_report'])
    
    print(f"Salvando dados em {output_file}...")
    final_df.to_parquet(output_file, index=False)
    print("Download de fundamentos concluído com sucesso!")

if __name__ == "__main__":
    fetch_fundamentals()
