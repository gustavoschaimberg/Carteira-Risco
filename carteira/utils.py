import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def mapear_para_pregao(datas, idx_precos):
    """
    Mapeia cada data para o primeiro pregão >= data no calendário de preços.

    Evita que movimentações em datas fora do calendário (fim de semana,
    feriado, ou anteriores ao primeiro pregão disponível) sejam descartadas
    silenciosamente por um reindex.

    Datas posteriores ao último pregão são alocadas no último pregão
    disponível, com warning.

    Parâmetros
    ----------
    datas      : Series/array de datas normalizadas.
    idx_precos : DatetimeIndex ordenado do histórico de preços.

    Retorna
    -------
    DatetimeIndex com as datas realocadas, mesmo comprimento de `datas`.
    """
    datas_arr = pd.DatetimeIndex(datas)
    pos = idx_precos.searchsorted(datas_arr, side="left")

    apos_ultimo = pos >= len(idx_precos)
    if apos_ultimo.any():
        n = int(apos_ultimo.sum())
        logger.warning(
            f"{n} movimentação(ões) posterior(es) ao último pregão disponível; "
            f"alocadas em {idx_precos[-1].date()}."
        )

    pos = np.clip(pos, 0, len(idx_precos) - 1)
    mapeadas = idx_precos[pos]

    realocadas = int((pd.DatetimeIndex(mapeadas) != datas_arr).sum())
    if realocadas > 0:
        logger.info(
            f"{realocadas} movimentação(ões) em datas sem pregão realocada(s) "
            f"para o pregão seguinte."
        )

    return pd.DatetimeIndex(mapeadas)


def extrair_float(valor):
    """Extrai um float de um escalar, Series ou array numpy."""
    if isinstance(valor, pd.Series):
        if valor.empty:
            raise ValueError("Série vazia ao tentar extrair preço.")
        return float(valor.iloc[0])
    return float(valor)


def remover_timezone(index):
    """Remove timezone de um DatetimeIndex de forma segura."""
    if getattr(index, "tz", None) is not None:
        return index.tz_convert(None)
    return index


def calcular_posicao_acumulada(grupo):
    """Retorna uma Series com a posição líquida acumulada por data."""
    sinal = grupo["tipo"].map({"COMPRA": 1, "VENDA": -1})
    return (grupo["quantidade"] * sinal).cumsum()
