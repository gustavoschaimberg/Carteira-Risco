import logging

import pandas as pd

logger = logging.getLogger(__name__)


def carregar_movimentacoes(arquivo):
    """
    Lê e valida a planilha de movimentações.
    Espera colunas: ticker, data, tipo, quantidade, preco.
    """

    logger.info(f"Carregando movimentações de: {arquivo}")

    df = pd.read_excel(arquivo)

    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["data"] = pd.to_datetime(df["data"], dayfirst=True, errors="coerce")
    df["tipo"] = df["tipo"].astype(str).str.strip().str.upper()
    df["quantidade"] = pd.to_numeric(df["quantidade"], errors="coerce")
    df["preco"] = pd.to_numeric(df["preco"], errors="coerce")

    linhas_antes = len(df)
    df = df.dropna(subset=["ticker", "data", "tipo", "quantidade", "preco"]).copy()
    df = df[df["tipo"].isin(["COMPRA", "VENDA"])].copy()
    df = df.sort_values(["ticker", "data"]).reset_index(drop=True)

    linhas_removidas = linhas_antes - len(df)
    if linhas_removidas > 0:
        logger.warning(f"{linhas_removidas} linha(s) ignoradas por dados inválidos ou tipo desconhecido.")

    if df.empty:
        raise ValueError("A planilha está vazia ou com colunas inválidas.")

    logger.info(f"{len(df)} movimentações carregadas para {df['ticker'].nunique()} ticker(s).")
    return df
