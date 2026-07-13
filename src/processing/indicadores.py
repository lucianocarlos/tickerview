import pandas as pd
import numpy as np

def calcular_PL(market_cap, lucro_liquido):
    """
    Calcula o múltiplo Preço sobre Lucro (P/L).
    Como proxy, usando Market Cap dividido por Lucro Líquido total.
    """
    return market_cap / lucro_liquido.replace(0, np.nan)

def calcular_PVP(market_cap, patrimonio_liquido):
    """
    Calcula o múltiplo Preço sobre Valor Patrimonial (P/VP).
    Proxy: Market Cap dividido pelo Patrimônio Líquido.
    """
    return market_cap / patrimonio_liquido.replace(0, np.nan)

def calcular_ROE(lucro_liquido, patrimonio_liquido):
    """
    Calcula o Return on Equity (ROE) anual.
    ROE = Lucro Líquido / Patrimônio Líquido
    """
    return lucro_liquido / patrimonio_liquido.replace(0, np.nan)

def calcular_DY(dividendos_pagos, market_cap):
    """
    Calcula o Dividend Yield (DY).
    DY = Total de Dividendos Pagos nos últimos 12 meses / Market Cap atual
    Atenção: dividendos_pagos deve ser uma soma rolling anual antes de ser passado para cá.
    """
    return dividendos_pagos / market_cap.replace(0, np.nan)

def calcular_Volatilidade(serie_precos, janela=21):
    """
    Calcula a volatilidade (desvio padrão dos retornos) em uma janela móvel (padrão de 21 dias).
    O resultado é anualizado assumindo 252 dias úteis.
    """
    retornos = serie_precos.pct_change()
    return retornos.rolling(window=janela).std() * np.sqrt(252)

def calcular_Retorno_Diario(serie_precos):
    """
    Calcula o retorno percentual diário básico de uma série de preços.
    """
    return serie_precos.pct_change()

def mesclar_cvm_precos(df_prices, df_cvm):
    """
    Realiza o alinhamento correto das séries diárias de preços com os balanços anuais da CVM.
    Usa merge_asof (backward fill temporal) para garantir que um indicador contábil do ano passado
    seja repetido diariamente até a publicação do balanço seguinte, impedindo Look-Ahead Bias.
    """
    df_prices = df_prices.copy()
    df_cvm = df_cvm.copy()
    
    # Padronização de nomes de coluna de data
    if 'Date' in df_prices.columns:
        df_prices = df_prices.rename(columns={'Date': 'data'})
    elif 'date' in df_prices.columns:
        df_prices = df_prices.rename(columns={'date': 'data'})
        
    if 'data_referencia' in df_cvm.columns:
        df_cvm = df_cvm.rename(columns={'data_referencia': 'data'})
        
    # Garantia de datetime
    df_prices['data'] = pd.to_datetime(df_prices['data'])
    df_cvm['data'] = pd.to_datetime(df_cvm['data'])
    
    # merge_asof exige bases ordenadas pela chave de tempo
    df_prices = df_prices.sort_values('data')
    df_cvm = df_cvm.sort_values('data')
    
    # Realiza o join guiado pelo tempo (asof) associado a cada ticker
    df_merged = pd.merge_asof(
        df_prices,
        df_cvm,
        on='data',
        by='ticker',
        direction='backward'
    )
    
    return df_merged
