import logging

import numpy as np
import pandas as pd

from carteira.market_data import baixar_dividendos
from carteira.utils import calcular_posicao_acumulada, extrair_float, mapear_para_pregao

logger = logging.getLogger(__name__)


def calcular_dividendos_recebidos(ticker, grupo):
    dividendos = baixar_dividendos(ticker)

    if dividendos.empty:
        return 0.0

    grupo_aux = grupo.copy()
    grupo_aux["data_norm"] = grupo_aux["data"].dt.normalize()
    grupo_aux["qtd_liquida"] = calcular_posicao_acumulada(grupo_aux)

    div_df = dividendos.reset_index()
    div_df.columns = ["data_div", "valor_div"]
    div_df["data_div"] = (
        pd.to_datetime(div_df["data_div"]).dt.normalize().astype("datetime64[us]")
    )
    div_df = div_df.sort_values("data_div")

    posicao_ref = (
        grupo_aux[["data_norm", "qtd_liquida"]]
        .drop_duplicates(subset="data_norm", keep="last")
        .sort_values("data_norm")
        .copy()
    )
    posicao_ref["data_norm"] = posicao_ref["data_norm"].astype("datetime64[us]")

    merged = pd.merge_asof(
        div_df,
        posicao_ref.rename(columns={"data_norm": "data_div"}),
        on="data_div",
        direction="backward",
    )

    merged["qtd_liquida"] = merged["qtd_liquida"].fillna(0.0)
    merged = merged[merged["qtd_liquida"] > 0]

    total = float((merged["valor_div"] * merged["qtd_liquida"]).sum())
    logger.debug(f"{ticker}: dividendos recebidos = R$ {total:.2f}")
    return total


def processar_ativo(ticker, grupo, precos_recentes):
    grupo = grupo.sort_values("data").reset_index(drop=True)

    quantidade_em_carteira = 0.0
    custo_total = 0.0
    capital_investido_total = 0.0
    lucro_realizado = 0.0
    resultados_movs = []

    for _, linha in grupo.iterrows():
        data_mov = pd.Timestamp(linha["data"]).normalize()
        tipo = linha["tipo"]
        quantidade = float(linha["quantidade"])
        preco = float(linha["preco"])

        if tipo == "COMPRA":
            valor_compra = quantidade * preco
            custo_total += valor_compra
            quantidade_em_carteira += quantidade
            capital_investido_total += valor_compra
            lucro_realizado_mov = 0.0
            financeiro = valor_compra

        elif tipo == "VENDA":
            if quantidade > quantidade_em_carteira:
                raise ValueError(
                    f"Venda maior que a posição em {ticker} na data {data_mov.date()}."
                )

            preco_medio = (
                custo_total / quantidade_em_carteira
                if quantidade_em_carteira > 0
                else 0.0
            )
            custo_baixado = quantidade * preco_medio
            valor_venda = quantidade * preco
            lucro_realizado_mov = valor_venda - custo_baixado

            lucro_realizado += lucro_realizado_mov
            custo_total -= custo_baixado
            quantidade_em_carteira -= quantidade

            if abs(quantidade_em_carteira) < 1e-9:
                quantidade_em_carteira = 0.0
                custo_total = 0.0

            financeiro = valor_venda

        else:
            continue

        resultados_movs.append({
            "ticker": ticker,
            "data": data_mov.date(),
            "tipo": tipo,
            "quantidade": quantidade,
            "preco": preco,
            "financeiro": financeiro,
            "quantidade_posicao": quantidade_em_carteira,
            "custo_total_posicao": custo_total,
            "preco_medio_posicao": (
                custo_total / quantidade_em_carteira
                if quantidade_em_carteira > 0
                else 0.0
            ),
            "lucro_realizado_mov": lucro_realizado_mov,
        })

    dividendos_recebidos_total = calcular_dividendos_recebidos(ticker, grupo)

    if quantidade_em_carteira > 0:
        if precos_recentes.empty or ticker not in precos_recentes.columns:
            raise ValueError(f"Não foi possível obter preço recente para {ticker}.")

        close_ticker = precos_recentes[ticker].dropna()
        if close_ticker.empty:
            raise ValueError(f"Sem cotações recentes para {ticker}.")

        preco_atual = extrair_float(close_ticker.iloc[[-1]])
        valor_atual = quantidade_em_carteira * preco_atual
        pnl_nao_realizado = valor_atual - custo_total
        retorno_pct_sem_div = pnl_nao_realizado / custo_total if custo_total != 0 else 0.0
        pnl_total_com_div = lucro_realizado + pnl_nao_realizado + dividendos_recebidos_total
        retorno_pct_com_div = (
            pnl_total_com_div / capital_investido_total
            if capital_investido_total != 0
            else 0.0
        )
        preco_medio = custo_total / quantidade_em_carteira

    else:
        preco_atual = 0.0
        valor_atual = 0.0
        pnl_nao_realizado = 0.0
        retorno_pct_sem_div = 0.0
        pnl_total_com_div = lucro_realizado + dividendos_recebidos_total
        retorno_pct_com_div = (
            pnl_total_com_div / capital_investido_total
            if capital_investido_total != 0
            else 0.0
        )
        preco_medio = 0.0

    resultado_ativo = {
        "ticker": ticker,
        "quantidade_total": quantidade_em_carteira,
        "preco_medio_compra": preco_medio,
        "preco_atual": preco_atual,
        "valor_investido_aberto": custo_total,
        "valor_atual": valor_atual,
        "dividendos_recebidos": dividendos_recebidos_total,
        "lucro_realizado": lucro_realizado,
        "pnl_nao_realizado": pnl_nao_realizado,
        "pnl_total_com_div": pnl_total_com_div,
        "retorno_pct_sem_div": retorno_pct_sem_div,
        "retorno_pct_com_div": retorno_pct_com_div,
        "capital_investido_total": capital_investido_total,
    }

    return resultados_movs, resultado_ativo


def consolidar_carteira(df, precos_recentes):
    resultados_movs = []
    resultados_ativos = []
    ativos_com_erro = []

    for ticker, grupo in df.groupby("ticker"):
        try:
            movs_ticker, ativo_ticker = processar_ativo(ticker, grupo, precos_recentes)
            resultados_movs.extend(movs_ticker)
            resultados_ativos.append(ativo_ticker)
            logger.info(f"{ticker}: processado com sucesso.")
        except Exception as e:
            logger.warning(f"{ticker}: erro ao processar — {e}. Ativo ignorado.")
            ativos_com_erro.append(ticker)

    if ativos_com_erro:
        logger.warning(f"Ativos ignorados por erro: {ativos_com_erro}")

    if not resultados_ativos:
        raise RuntimeError("Nenhum ativo pôde ser processado. Verifique os dados e os logs.")

    return pd.DataFrame(resultados_movs), pd.DataFrame(resultados_ativos)


def gerar_resumo(df_ativos):
    if df_ativos.empty:
        return pd.DataFrame()

    capital_total_geral = df_ativos["capital_investido_total"].sum()

    resumo = {
        "total_investido_aberto": df_ativos["valor_investido_aberto"].sum(),
        "total_atual": df_ativos["valor_atual"].sum(),
        "total_dividendos_recebidos": df_ativos["dividendos_recebidos"].sum(),
        "total_lucro_realizado": df_ativos["lucro_realizado"].sum(),
        "total_pnl_nao_realizado": df_ativos["pnl_nao_realizado"].sum(),
        "total_pnl_com_dividendos": df_ativos["pnl_total_com_div"].sum(),
        "quantidade_ativos": df_ativos[df_ativos["quantidade_total"] > 0]["ticker"].nunique(),
    }

    resumo["retorno_total_pct"] = (
        resumo["total_pnl_com_dividendos"] / capital_total_geral
        if capital_total_geral != 0
        else 0.0
    )

    return pd.DataFrame([resumo])


def calcular_alpha_temporal_por_ativo(df_movimentacoes, precos_hist, df_ibov):
    """
    Calcula contribuição de alpha por ativo usando séries temporais diárias.

    Lógica:
    - calcula posição diária de cada ativo
    - calcula peso diário de cada ativo na carteira
    - calcula retorno diário de cada ativo
    - compara contra retorno diário do IBOV no mesmo dia
    - soma as contribuições diárias de alpha por ativo

    Essa versão mantém o mesmo padrão temporal para ativo e benchmark.
    """

    if df_movimentacoes.empty or precos_hist.empty or df_ibov.empty:
        return pd.DataFrame()

    df = df_movimentacoes.copy()
    df["data_norm"] = pd.to_datetime(df["data"]).dt.normalize()
    # Realoca movimentações fora do calendário de preços para o pregão
    # seguinte — sem isso, o reindex abaixo as descartaria silenciosamente.
    df["data_ref"] = mapear_para_pregao(df["data_norm"], precos_hist.index)
    df["qtd_sinal"] = df.apply(
        lambda r: r["quantidade"] if r["tipo"] == "COMPRA" else -r["quantidade"],
        axis=1,
    )

    # =====================================================
    # POSIÇÕES DIÁRIAS
    # =====================================================
    posicoes = (
        df.groupby(["data_ref", "ticker"])["qtd_sinal"]
        .sum()
        .unstack()
        .reindex(precos_hist.index)
        .fillna(0.0)
        .cumsum()
    )

    # Posição negativa indica erro nos dados de entrada (venda além da
    # posição) — loga em vez de mascarar silenciosamente.
    negativos = (posicoes < -1e-9).any()
    if negativos.any():
        tickers_neg = list(negativos[negativos].index)
        logger.warning(
            f"Posição negativa detectada em {tickers_neg} — verifique as "
            f"movimentações. Valores truncados em zero para o cálculo."
        )
    posicoes = posicoes.clip(lower=0)

    tickers = [t for t in posicoes.columns if t in precos_hist.columns]

    if not tickers:
        return pd.DataFrame()

    precos = precos_hist[tickers].ffill()
    posicoes = posicoes[tickers]

    valores = posicoes * precos
    valor_total = valores.sum(axis=1).replace(0, np.nan)

    pesos = valores.div(valor_total, axis=0).fillna(0.0)
    pesos_ontem = pesos.shift(1).fillna(0.0)

    retornos_ativos = precos.pct_change().fillna(0.0)

    ibov = df_ibov["IBOV"].reindex(precos.index).ffill()
    retorno_ibov = ibov.pct_change().fillna(0.0)

    contrib_retorno = pesos_ontem * retornos_ativos
    contrib_benchmark = pesos_ontem.mul(retorno_ibov, axis=0)
    contrib_alpha = contrib_retorno - contrib_benchmark

    resultado = pd.DataFrame({
        "ticker": tickers,
        "peso_atual": pesos.iloc[-1][tickers].values,
        "peso_medio": pesos_ontem[tickers].mean().values,
        "contrib_retorno_carteira": contrib_retorno[tickers].sum().values,
        "contrib_benchmark_ibov": contrib_benchmark[tickers].sum().values,
        "contrib_alpha": contrib_alpha[tickers].sum().values,
        "dias_com_posicao": (posicoes[tickers] > 0).sum().values,
    })

    resultado["alpha_individual_aprox"] = np.where(
        resultado["peso_medio"] > 0.01,
        resultado["contrib_alpha"] / resultado["peso_medio"],
        np.nan,
    )

    return resultado.sort_values("contrib_alpha")