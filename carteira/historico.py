import logging

import pandas as pd

from carteira.utils import mapear_para_pregao

logger = logging.getLogger(__name__)

_EPS = 1e-9


def _calcular_caixa_dividendos(posicao_pivot, dividendos_por_ticker, idx_precos):
    """
    Constrói a série diária de caixa acumulado de dividendos.

    O direito ao provento é da posição no fechamento do dia anterior
    ao ex-date (posição "com" direito), por isso usa shift(1).
    O crédito entra no ex-date — mesmo dia em que o preço cai —
    mantendo a cota como medida de retorno total.
    """
    caixa = pd.Series(0.0, index=idx_precos)

    if not dividendos_por_ticker:
        return caixa

    for ticker, divs in dividendos_por_ticker.items():
        if divs is None or divs.empty or ticker not in posicao_pivot.columns:
            continue

        divs_periodo = divs[(divs.index >= idx_precos[0]) & (divs.index <= idx_precos[-1])]
        if divs_periodo.empty:
            continue

        # ex-dates do yfinance são pregões; se algum não casar com o
        # calendário (fonte de preços diferente), realoca para o seguinte
        datas_ex = mapear_para_pregao(divs_periodo.index, idx_precos)
        valores = pd.Series(divs_periodo.values, index=datas_ex).groupby(level=0).sum()

        posicao_d1 = posicao_pivot[ticker].shift(1).fillna(0.0)
        credito = (valores.reindex(idx_precos).fillna(0.0) * posicao_d1).clip(lower=0.0)

        total = float(credito.sum())
        if total > 0:
            logger.debug(f"{ticker}: R$ {total:.2f} em dividendos creditados na cota.")

        caixa += credito

    return caixa.cumsum()


def construir_historico_cotizado(df, precos_hist, dividendos_por_ticker=None):
    """
    Constrói o histórico cotizado da carteira (retorno total).

    Convenções
    ----------
    - Fluxos convertem em cotas pela cota de fechamento do dia anterior (D-1).
      A diferença entre preço de execução e fechamento aparece como
      performance do dia — inclusive no primeiro dia.
    - Movimentações em datas fora do calendário de preços são realocadas
      para o primeiro pregão seguinte (nada é descartado).
    - Dividendos entram como caixa acumulado no patrimônio a partir do
      ex-date (posição de D-1 define o direito). Com preços não ajustados,
      a queda do preço no ex-date é compensada pelo crédito em caixa,
      tornando a cota uma série de retorno total.
    - Liquidação total sem caixa residual: a cota do dia é marcada pelo
      valor efetivamente realizado na venda e as cotas zeram. Reentradas
      posteriores convertem pela última cota, preservando a continuidade.
    """

    colunas = ["data", "fluxo_do_dia", "valor_mercado", "cotas", "cota", "caixa_dividendos"]

    if df.empty or precos_hist.empty:
        logger.warning("Histórico cotizado não pôde ser construído: dados insuficientes.")
        return pd.DataFrame(columns=colunas)

    idx_precos = precos_hist.index

    df_pos = df.copy()
    df_pos["data_norm"] = df_pos["data"].dt.normalize()
    df_pos["data_ref"] = mapear_para_pregao(df_pos["data_norm"], idx_precos)
    df_pos["qtd_sinal"] = df_pos.apply(
        lambda r: r["quantidade"] if r["tipo"] == "COMPRA" else -r["quantidade"],
        axis=1,
    )
    df_pos["fluxo"] = df_pos.apply(
        lambda r: r["quantidade"] * r["preco"]
        if r["tipo"] == "COMPRA"
        else -r["quantidade"] * r["preco"],
        axis=1,
    )

    # Como data_ref pertence ao calendário de preços, o reindex não
    # descarta nenhuma movimentação antes do cumsum.
    posicao_pivot = (
        df_pos.groupby(["data_ref", "ticker"])["qtd_sinal"]
        .sum()
        .unstack()
        .reindex(idx_precos)
        .fillna(0.0)
        .cumsum()
    )

    fluxo_pivot = (
        df_pos.groupby("data_ref")["fluxo"]
        .sum()
        .reindex(idx_precos)
        .fillna(0.0)
    )

    tickers = sorted(df["ticker"].unique().tolist())
    tickers_comuns = [
        t for t in tickers
        if t in posicao_pivot.columns and t in precos_hist.columns
    ]

    tickers_ausentes = set(tickers) - set(tickers_comuns)
    if tickers_ausentes:
        logger.warning(f"Tickers sem preço histórico, ignorados no histórico: {tickers_ausentes}")

    valor_acoes_diario = (
        posicao_pivot[tickers_comuns] * precos_hist[tickers_comuns].ffill()
    ).sum(axis=1)

    caixa_div_acum = _calcular_caixa_dividendos(
        posicao_pivot[tickers_comuns], dividendos_por_ticker, idx_precos
    )

    patrimonio_diario = valor_acoes_diario + caixa_div_acum

    historico_carteira = []
    cotas_totais = 0.0
    cota = 1.0
    iniciado = False

    for data_ref in idx_precos:
        pl = float(patrimonio_diario.get(data_ref, 0.0))
        fluxo_do_dia = float(fluxo_pivot.get(data_ref, 0.0))
        caixa_hoje = float(caixa_div_acum.get(data_ref, 0.0))

        if not iniciado:
            if fluxo_do_dia <= 0:
                continue
            # Primeiro aporte: cotas emitidas a cota = 1.0
            cotas_totais = fluxo_do_dia
            iniciado = True
            if pl > 0:
                cota = pl / cotas_totais

        elif pl <= _EPS and fluxo_do_dia < 0 and cotas_totais > _EPS:
            # Liquidação total sem caixa residual: marca a cota pelo
            # valor realizado e zera as cotas. A cota sobrevive para
            # uma eventual reentrada.
            cota = -fluxo_do_dia / cotas_totais
            cotas_totais = 0.0

        else:
            if fluxo_do_dia != 0:
                cotas_totais += fluxo_do_dia / cota
            if cotas_totais > _EPS and pl > 0:
                cota = pl / cotas_totais
            elif cotas_totais <= _EPS:
                cotas_totais = 0.0

        historico_carteira.append({
            "data": data_ref,
            "fluxo_do_dia": fluxo_do_dia,
            "valor_mercado": pl,
            "cotas": cotas_totais,
            "cota": cota,
            "caixa_dividendos": caixa_hoje,
        })

    df_result = pd.DataFrame(historico_carteira, columns=colunas)
    logger.info(f"Histórico cotizado construído: {len(df_result)} dias úteis.")
    return df_result
