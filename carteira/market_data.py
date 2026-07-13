import logging

import pandas as pd
import yfinance as yf

from carteira.utils import remover_timezone

logger = logging.getLogger(__name__)


def _normalizar_precos(precos_brutos, tickers):
    """Normaliza o DataFrame retornado pelo yfinance para índice=data, colunas=tickers."""
    if isinstance(precos_brutos.columns, pd.MultiIndex):
        precos = precos_brutos["Close"].copy()
    else:
        precos = precos_brutos[["Close"]].copy()
        precos.columns = [tickers[0]]

    if isinstance(precos, pd.Series):
        precos = precos.to_frame(name=tickers[0])

    precos.index = remover_timezone(precos.index)
    precos.index = pd.to_datetime(precos.index).normalize()
    precos = precos.sort_index().ffill()
    return precos


def baixar_precos_recentes(tickers, period="5d"):
    """
    Baixa preços recentes de fechamento para todos os tickers.
    Retorna DataFrame com índice=data e colunas=tickers.
    """

    logger.info(f"Baixando preços recentes para {len(tickers)} ticker(s)...")

    precos_brutos = yf.download(
        tickers,
        period=period,
        auto_adjust=False,
        progress=False,
    )

    if precos_brutos.empty:
        logger.warning("Nenhum preço recente retornado pelo yfinance.")
        return pd.DataFrame()

    return _normalizar_precos(precos_brutos, tickers)


def baixar_historico_precos(tickers, data_inicio, data_fim):
    """
    Baixa histórico de preços de fechamento.
    Retorna DataFrame com índice=data e colunas=tickers.
    """

    logger.info("Baixando histórico de preços para o gráfico de cotas...")

    precos_brutos = yf.download(
        tickers,
        start=data_inicio.strftime("%Y-%m-%d"),
        end=(data_fim + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
        group_by="column",
    )

    if precos_brutos.empty:
        logger.warning("Nenhum histórico de preços retornado pelo yfinance.")
        return pd.DataFrame()

    return _normalizar_precos(precos_brutos, tickers)

# Cache em memória: evita baixar os mesmos dividendos duas vezes
# (consolidar_carteira e construir_historico_cotizado usam a mesma série)
_CACHE_DIVIDENDOS: dict[str, pd.Series] = {}


def baixar_dividendos(ticker):
    """Baixa dividendos de um ticker via yfinance (com cache em memória)."""

    if ticker in _CACHE_DIVIDENDOS:
        return _CACHE_DIVIDENDOS[ticker].copy()

    acao = yf.Ticker(ticker)
    dividendos = acao.dividends.copy()

    if not dividendos.empty:
        dividendos.index = remover_timezone(dividendos.index)
        dividendos.index = pd.to_datetime(dividendos.index).normalize()
        logger.debug(f"{ticker}: {len(dividendos)} evento(s) de dividendo encontrado(s).")
    else:
        logger.debug(f"{ticker}: sem dividendos registrados.")

    _CACHE_DIVIDENDOS[ticker] = dividendos.copy()
    return dividendos

def baixar_benchmark_ibov(data_inicio, data_fim):
    """
    Baixa histórico do IBOV (^BVSP) via Yahoo Finance.
    Retorna DataFrame com coluna 'IBOV'.
    """

    logger.info("Baixando benchmark IBOV (^BVSP)...")

    df = yf.download(
        "^BVSP",
        start=data_inicio.strftime("%Y-%m-%d"),
        end=(data_fim + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        logger.warning("IBOV não retornou dados.")
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        if "Adj Close" in df.columns.get_level_values(0):
            df_ibov = df["Adj Close"].copy()
        else:
            df_ibov = df["Close"].copy()
    else:
        if "Adj Close" in df.columns:
            df_ibov = df[["Adj Close"]].copy()
        else:
            df_ibov = df[["Close"]].copy()

    if isinstance(df_ibov, pd.Series):
        df_ibov = df_ibov.to_frame(name="IBOV")
    else:
        df_ibov.columns = ["IBOV"]

    df_ibov.index = remover_timezone(df_ibov.index)
    df_ibov.index = pd.to_datetime(df_ibov.index).normalize()
    df_ibov = df_ibov.sort_index().ffill()

    return df_ibov