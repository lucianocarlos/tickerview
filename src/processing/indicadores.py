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

def calcular_Momentum(serie_precos, janela=5):
    """
    Calcula o momentum como a variação percentual do preço em uma dada janela.
    """
    return serie_precos.pct_change(periods=janela)

def calcular_Anomalia_Volume(serie_volume, janela=21):
    """
    Volume / Média Móvel Simples de Volume (confirmação de tendência)
    """
    volume_ma = serie_volume.rolling(window=janela).mean()
    return serie_volume / volume_ma.replace(0, np.nan)

def calcular_RSI(serie_precos, janela=14):
    """
    Relative Strength Index (RSI) clássico.
    """
    delta = serie_precos.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    
    # EMA
    ma_up = up.ewm(com=janela-1, adjust=False, min_periods=janela).mean()
    ma_down = down.ewm(com=janela-1, adjust=False, min_periods=janela).mean()
    
    rs = ma_up / ma_down.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    # Para ma_down == 0, RSI deveria ser 100
    rsi[ma_down == 0] = 100
    return rsi

def calcular_MACD(serie_precos, rapida=12, lenta=26, sinal=9):
    """
    Calcula a linha do MACD, a Linha de Sinal e o Histograma.
    Retorna apenas o Histograma (MACD - Sinal) como feature de momento direcional.
    """
    ema_rapida = serie_precos.ewm(span=rapida, adjust=False).mean()
    ema_lenta = serie_precos.ewm(span=lenta, adjust=False).mean()
    macd_line = ema_rapida - ema_lenta
    signal_line = macd_line.ewm(span=sinal, adjust=False).mean()
    histograma = macd_line - signal_line
    return histograma

def calcular_Bollinger_Width(serie_precos, janela=21, k=2):
    """
    Largura das Bandas de Bollinger normalizada pelo preço.
    BBW = (Upper - Lower) / SMA
    """
    sma = serie_precos.rolling(window=janela).mean()
    std = serie_precos.rolling(window=janela).std()
    upper = sma + (k * std)
    lower = sma - (k * std)
    return (upper - lower) / sma.replace(0, np.nan)

def calcular_Alavancagem(divida_total, ativo_total):
    """
    Dívida Total / Ativo Total
    """
    return divida_total / ativo_total.replace(0, np.nan)

def calcular_Margem_EBIT(ebit, receita_total):
    """
    EBIT / Receita Total
    """
    return ebit / receita_total.replace(0, np.nan)

def calcular_ZScore_Setorial(df, coluna_alvo, coluna_setor='sector'):
    """
    Calcula o Z-Score cross-sectional de uma coluna por setor para uma data específica.
    OBS: Deve ser aplicado agrupando por 'Date' ou 'data_referencia'.
    """
    def zscore(x):
        std = x.std()
        if pd.isna(std) or std == 0:
            return pd.Series(0, index=x.index)
        return (x - x.mean()) / std
    return df.groupby(coluna_setor)[coluna_alvo].transform(zscore)

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
