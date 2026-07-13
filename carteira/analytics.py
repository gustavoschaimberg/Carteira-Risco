import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf

logger = logging.getLogger(__name__)

# =========================================================
# CONSTANTES SGS/BCB
# Endpoint com filtro de datas — obrigatório desde mar/2025
# =========================================================
_SGS_SELIC_DIARIA = 11   # Selic diária (% a.d.)
_SGS_IPCA_MENSAL  = 433  # IPCA mensal (% a.m.)
_TIMEOUT          = 10
_FALLBACK_SELIC   = 0.105  # 10,5% a.a. — atualizar se necessário

# Ticker do IBOV no yfinance
_TICKER_IBOV = "^BVSP"


# =========================================================
# FUNÇÕES DE BUSCA — BCB/SGS
# =========================================================

def _buscar_sgs(codigo: int, n: int) -> pd.DataFrame | None:
    """
    Busca série SGS/BCB usando intervalo de datas (evita erro 406
    do endpoint /ultimos em séries longas).
    Retorna os últimos N registros ou None se falhar.
    """
    data_final   = date.today()
    data_inicial = data_final - timedelta(days=500)

    url = (
        f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados"
        f"?formato=json"
        f"&dataInicial={data_inicial.strftime('%d/%m/%Y')}"
        f"&dataFinal={data_final.strftime('%d/%m/%Y')}"
    )

    try:
        resp = requests.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        dados = resp.json()

        if not dados:
            logger.warning(f"SGS {codigo}: resposta vazia.")
            return None

        df = pd.DataFrame(dados)
        df["data"]  = pd.to_datetime(df["data"], dayfirst=True)
        df["valor"] = pd.to_numeric(df["valor"], errors="coerce")
        df = df.dropna(subset=["valor"]).sort_values("data").tail(n)

        return df

    except Exception as e:
        logger.warning(f"SGS {codigo}: falha na requisição — {e}")
        return None


def buscar_taxa_bcb_selic() -> tuple[float, str]:
    """
    Busca a Selic diária (SGS 11) dos últimos 252 dias úteis,
    acumula e anualiza.
    Levanta RuntimeError se não conseguir obter os dados.
    """
    df = _buscar_sgs(_SGS_SELIC_DIARIA, n=252)

    if df is None or df.empty:
        raise RuntimeError("BCB/SGS indisponível para Selic diária.")

    taxa_acumulada = (1 + df["valor"] / 100).prod() - 1
    taxa_anual     = (1 + taxa_acumulada) ** (252 / len(df)) - 1

    logger.info(f"Selic via BCB (últimos {len(df)} dias úteis): {taxa_anual:.2%} a.a.")
    return taxa_anual, "BCB/Selic-diária"


def buscar_ipca_12m() -> float:
    """
    Busca o IPCA dos últimos 12 meses (SGS 433) e acumula.
    Retorna taxa decimal ou np.nan se falhar.
    """
    df = _buscar_sgs(_SGS_IPCA_MENSAL, n=12)

    if df is None or df.empty:
        logger.warning("IPCA 12m: não foi possível obter do BCB.")
        return np.nan

    ipca_12m = (1 + df["valor"] / 100).prod() - 1
    logger.info(f"IPCA 12m via BCB: {ipca_12m:.2%}")
    return float(ipca_12m)


# =========================================================
# BENCHMARK — IBOV
# =========================================================

def buscar_retorno_ibov(data_inicio: pd.Timestamp, data_fim: pd.Timestamp) -> pd.Series | None:
    """
    Baixa o histórico do IBOV no período da carteira via yfinance.
    Retorna Series com índice=data e valores=preço de fechamento, ou None.
    """
    try:
        dados = yf.download(
            _TICKER_IBOV,
            start=data_inicio.strftime("%Y-%m-%d"),
            end=(data_fim + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )
        if dados.empty:
            logger.warning("IBOV: sem dados retornados pelo yfinance.")
            return None

        close = dados["Close"].squeeze()
        close.index = pd.to_datetime(close.index).normalize()
        return close.sort_index()

    except Exception as e:
        logger.warning(f"IBOV: falha ao baixar via yfinance — {e}")
        return None


# =========================================================
# MÉTRICAS DE BETA / CAPM
# =========================================================

def _calcular_beta_e_capm(
    retornos_carteira: pd.Series,
    retornos_ibov: pd.Series,
    rf_diario: float,
    retorno_anualizado_carteira: float,
    retorno_anualizado_ibov: float,
) -> dict:
    """
    Calcula beta, Treynor ratio e Jensen's alpha (CAPM alpha).

    Fórmulas
    --------
    Beta         = Cov(Rc, Rm) / Var(Rm)
    Treynor      = (Rc_anual - Rf_anual) / Beta
    Jensen alpha = Rc_anual - [Rf_anual + Beta * (Rm_anual - Rf_anual)]
    """
    nulos = {
        "beta_ibov":    np.nan,
        "treynor":      np.nan,
        "jensen_alpha": np.nan,
    }

    df_par = pd.DataFrame({
        "carteira": retornos_carteira,
        "ibov":     retornos_ibov,
    }).dropna()

    if len(df_par) < 20:
        logger.warning("Beta: menos de 20 observações comuns com IBOV. Métricas não calculadas.")
        return nulos

    cov_matrix = df_par.cov()
    var_ibov   = df_par["ibov"].var()

    if var_ibov == 0:
        logger.warning("Beta: variância do IBOV é zero. Métricas não calculadas.")
        return nulos

    beta         = cov_matrix.loc["carteira", "ibov"] / var_ibov
    rfr_anual    = (1 + rf_diario) ** 252 - 1
    premio_ibov  = retorno_anualizado_ibov - rfr_anual

    treynor      = (retorno_anualizado_carteira - rfr_anual) / beta if beta != 0 else np.nan
    jensen_alpha = retorno_anualizado_carteira - (rfr_anual + beta * premio_ibov)

    logger.info(
        f"Beta IBOV: {beta:.3f} | "
        f"Treynor: {treynor:.4f} | "
        f"Jensen alpha: {jensen_alpha:.2%}"
    )

    return {
        "beta_ibov":    beta,
        "treynor":      treynor,
        "jensen_alpha": jensen_alpha,
    }


# =========================================================
# DRAWDOWN ATUAL E TEMPO DE RECUPERAÇÃO
# =========================================================

def _calcular_drawdown_atual(df: pd.DataFrame) -> dict:
    """
    Calcula o drawdown atual e o tempo de recuperação estimado.

    drawdown_atual            : queda percentual vs o último pico.
    dias_em_drawdown          : dias úteis desde que o último pico foi atingido.
    recuperacao_estimada_dias : dias úteis necessários para recuperar o drawdown,
                                dado o retorno médio diário dos últimos 60 pregões.
                                np.nan quando a carteira está no pico ou o retorno
                                médio recente é negativo.
    """
    nulos = {
        "drawdown_atual":            np.nan,
        "dias_em_drawdown":          np.nan,
        "recuperacao_estimada_dias": np.nan,
    }

    if "drawdown" not in df.columns or "pico" not in df.columns:
        return nulos

    drawdown_atual = float(df["drawdown"].iloc[-1])
    pico_valor     = float(df["pico"].iloc[-1])

    # Índice posicional do último pico
    idx_ultimo_pico = df.index[df["cota"] >= pico_valor * (1 - 1e-9)].max()
    dias_em_drawdown = int(df.index[-1] - idx_ultimo_pico)

    # Retorno médio dos últimos 60 pregões para estimar recuperação
    retorno_medio_diario = df["retorno"].dropna().tail(60).mean()

    if drawdown_atual < -1e-6 and retorno_medio_diario > 1e-9:
        # (1 + r)^n = 1 / (1 + dd)  →  n = log(1/(1+dd)) / log(1+r)
        n = np.log(1 / (1 + drawdown_atual)) / np.log(1 + retorno_medio_diario)
        recuperacao_estimada = int(np.ceil(n))
    else:
        recuperacao_estimada = np.nan

    logger.info(
        f"Drawdown atual: {drawdown_atual:.2%} | "
        f"Dias em drawdown: {dias_em_drawdown} | "
        f"Recuperação estimada: {recuperacao_estimada} dias úteis"
    )

    return {
        "drawdown_atual":            drawdown_atual,
        "dias_em_drawdown":          dias_em_drawdown,
        "recuperacao_estimada_dias": recuperacao_estimada,
    }


# =========================================================
# MÉTRICAS DE RISCO — FUNÇÃO PRINCIPAL
# =========================================================

def calcular_metricas_risco(
    df_historico: pd.DataFrame,
    taxa_livre_risco_anual: float | None = None,
    janela_rolling: int = 60,
    ibov_close: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calcula métricas de risco usando a série de cotas da carteira.

    Parâmetros
    ----------
    df_historico : DataFrame com colunas 'data' e 'cota'.
    taxa_livre_risco_anual : Taxa livre de risco anual em decimal.
        - None (padrão): busca Selic real via SGS/BCB; fallback 10,5% a.a.
        - Qualquer float (inclusive 0.0): usa diretamente sem chamar a API.
    janela_rolling : Janela em dias úteis para volatilidade rolling.
    ibov_close : Série de fechamento do IBOV já baixada (índice=data).
        Se None, baixa internamente via yfinance (fallback). Passar a série
        evita download duplicado e garante que todas as métricas usem a
        mesma fonte de preços do restante do relatório.

    Retorna
    -------
    (df_metricas, df_historico_enriquecido)

    Métricas calculadas
    -------------------
    Básicas         : retorno acumulado, anualizado, real (vs IPCA), volatilidade
    Risco ajustado  : Sharpe, Sortino, Calmar, VaR 95%, CVaR 95%
    Drawdown        : máximo, atual, dias em drawdown, recuperação estimada
    Benchmark IBOV  : retorno acumulado/anualizado, alpha simples,
                      beta, Treynor, Jensen alpha (CAPM alpha)
    """

    if df_historico.empty or "cota" not in df_historico.columns:
        logger.warning("Histórico vazio ou sem coluna 'cota'. Métricas não calculadas.")
        return pd.DataFrame(), pd.DataFrame()

    # --- Taxa livre de risco ---
    if taxa_livre_risco_anual is not None:
        rfr   = taxa_livre_risco_anual
        fonte = f"usuário: {rfr:.2%} a.a."
        logger.info(f"Taxa livre de risco definida manualmente: {rfr:.2%}")
    else:
        try:
            rfr, fonte = buscar_taxa_bcb_selic()
        except Exception:
            rfr   = _FALLBACK_SELIC
            fonte = f"fallback {_FALLBACK_SELIC:.1%} a.a. (BCB indisponível)"
            logger.warning(f"BCB indisponível. Usando fallback {_FALLBACK_SELIC:.1%} a.a.")

    rf_diario = (1 + rfr) ** (1 / 252) - 1

    # --- Preparação da série ---
    df = df_historico.copy()
    df["data"] = pd.to_datetime(df["data"])
    df = df.sort_values("data").reset_index(drop=True)

    df["retorno"] = df["cota"].pct_change()
    retornos = df["retorno"].dropna()

    if retornos.empty:
        logger.warning("Série de retornos vazia. Métricas não calculadas.")
        return pd.DataFrame(), df

    # --- Retornos ---
    retorno_acumulado  = df["cota"].iloc[-1] / df["cota"].iloc[0] - 1
    dias               = (df["data"].iloc[-1] - df["data"].iloc[0]).days
    retorno_anualizado = (1 + retorno_acumulado) ** (365 / dias) - 1 if dias > 0 else 0.0

    # --- Risco ---
    volatilidade = retornos.std() * np.sqrt(252)

    excesso = retornos - rf_diario
    sharpe  = (
        excesso.mean() / retornos.std() * np.sqrt(252)
        if retornos.std() != 0 else np.nan
    )

    downside     = retornos[retornos < rf_diario] - rf_diario
    downside_dev = downside.std() * np.sqrt(252) if len(downside) > 1 else np.nan
    sortino      = (
        excesso.mean() * 252 / downside_dev
        if downside_dev and downside_dev != 0 else np.nan
    )

    # --- Drawdown ---
    df["pico"]     = df["cota"].cummax()
    df["drawdown"] = df["cota"] / df["pico"] - 1
    max_dd         = df["drawdown"].min()
    calmar         = retorno_anualizado / abs(max_dd) if max_dd != 0 else np.nan

    # --- VaR / CVaR ---
    var_95  = retornos.quantile(0.05)
    cvar_95 = retornos[retornos <= var_95].mean()

    # --- IPCA e retorno real ---
    ipca_12m     = buscar_ipca_12m()
    retorno_real = (
        (1 + retorno_anualizado) / (1 + ipca_12m) - 1
        if not np.isnan(ipca_12m) else np.nan
    )

    # --- Benchmark IBOV ---
    if ibov_close is None:
        ibov_close = buscar_retorno_ibov(df["data"].iloc[0], df["data"].iloc[-1])

    retorno_ibov_acumulado  = np.nan
    retorno_anualizado_ibov = np.nan
    alpha_vs_ibov           = np.nan
    metricas_beta           = {"beta_ibov": np.nan, "treynor": np.nan, "jensen_alpha": np.nan}

    if ibov_close is not None and len(ibov_close) > 0:
        # Alinhamento as-of: dias da carteira sem pregão coincidente do IBOV
        # recebem o último fechamento disponível, em vez de virarem NaN e
        # reduzirem as observações do beta.
        datas_carteira = pd.DatetimeIndex(df["data"]).normalize()
        df["ibov_close"] = (
            ibov_close.sort_index()
            .reindex(datas_carteira, method="ffill")
            .values
        )

        ibov_validos = df["ibov_close"].dropna()
        if len(ibov_validos) >= 2:
            retorno_ibov_acumulado  = ibov_validos.iloc[-1] / ibov_validos.iloc[0] - 1
            retorno_anualizado_ibov = (
                (1 + retorno_ibov_acumulado) ** (365 / dias) - 1 if dias > 0 else np.nan
            )
            alpha_vs_ibov = retorno_anualizado - retorno_anualizado_ibov

        retornos_ibov = df["ibov_close"].pct_change()
        retornos_ibov.index = df.index

        metricas_beta = _calcular_beta_e_capm(
            retornos_carteira           = retornos,
            retornos_ibov               = retornos_ibov,
            rf_diario                   = rf_diario,
            retorno_anualizado_carteira = retorno_anualizado,
            retorno_anualizado_ibov     = retorno_anualizado_ibov,
        )

    # --- Drawdown atual e recuperação ---
    metricas_drawdown = _calcular_drawdown_atual(df)

    # --- Rolling volatility ---
    df["rolling_vol"] = retornos.rolling(janela_rolling).std() * np.sqrt(252)

    logger.info(
        f"Métricas — Retorno: {retorno_acumulado:.2%} | "
        f"Sharpe: {sharpe:.2f} | Sortino: {sortino:.2f} | Calmar: {calmar:.2f} | "
        f"MaxDD: {max_dd:.2%} | RF: {rfr:.2%} ({fonte})"
    )

    metricas = pd.DataFrame([{
        # Taxa de referência
        "taxa_livre_risco_anual":   rfr,
        "fonte_taxa_livre_risco":   fonte,
        # Inflação
        "ipca_12m":                 ipca_12m,
        # Retornos
        "retorno_acumulado":        retorno_acumulado,
        "retorno_anualizado":       retorno_anualizado,
        "retorno_real_anualizado":  retorno_real,
        # Risco
        "volatilidade_anualizada":  volatilidade,
        # Índices ajustados ao risco
        "sharpe":                   sharpe,
        "sortino":                  sortino,
        "calmar":                   calmar,
        # Drawdown
        "max_drawdown":             max_dd,
        **metricas_drawdown,
        # Risco de cauda
        "var_95_diario":            var_95,
        "cvar_95_diario":           cvar_95,
        # Benchmark IBOV — retornos e alpha simples
        "retorno_ibov_acumulado":   retorno_ibov_acumulado,
        "retorno_ibov_anualizado":  retorno_anualizado_ibov,
        "alpha_vs_ibov":            alpha_vs_ibov,
        # Benchmark IBOV — CAPM
        **metricas_beta,
    }])

    return metricas, df


# =========================================================
# ALPHA E BETA POR ATIVO — CAPM
# =========================================================

def calcular_alpha_por_ativo_capm(
    precos_hist: pd.DataFrame,
    ibov_close: pd.Series | None,
    rf_diario: float,
    dias_carteira: int,
    df_movimentacoes: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Calcula beta, Jensen alpha (CAPM alpha), Treynor, Sharpe e volatilidade
    por ativo individual, usando os retornos diários do histórico de preços.

    O histórico de cada ativo começa na data da primeira compra,
    assim o cálculo reflete apenas o período em que o ativo estava na carteira.

    Parâmetros
    ----------
    precos_hist      : DataFrame com colunas = tickers e índice = datas.
    ibov_close       : Series com preços de fechamento do IBOV (mesmo índice).
    rf_diario        : Taxa livre de risco diária em decimal.
    dias_carteira    : Dias corridos da carteira (para anualizar retornos).
    df_movimentacoes : DataFrame com colunas 'ticker', 'data', 'tipo'.
                       Usado para determinar a data da primeira compra de cada ativo.

    Retorna
    -------
    DataFrame com uma linha por ticker, ordenado por alpha_capm_anualizado desc.
    Colunas: ticker, retorno_acumulado, retorno_anualizado, beta_ibov,
             alpha_capm_anualizado, treynor, volatilidade_anualizada,
             sharpe_individual, n_obs, data_primeira_compra, dias_na_carteira.
    """

    if precos_hist.empty or ibov_close is None or ibov_close.empty:
        logger.warning("Alpha por ativo CAPM: dados insuficientes.")
        return pd.DataFrame()

    rfr_anual = (1 + rf_diario) ** 252 - 1

    # Data da primeira compra por ticker
    data_primeira_compra = {}
    if df_movimentacoes is not None and not df_movimentacoes.empty:
        compras = df_movimentacoes[df_movimentacoes["tipo"] == "COMPRA"].copy()
        compras["data"] = pd.to_datetime(compras["data"])
        data_primeira_compra = compras.groupby("ticker")["data"].min().to_dict()

    resultados = []

    for ticker in precos_hist.columns:
        # Corta histórico a partir da primeira compra
        serie_full = precos_hist[ticker].ffill()
        data_ini   = None

        if ticker in data_primeira_compra:
            data_ini = pd.Timestamp(data_primeira_compra[ticker]).normalize()
            serie_full = serie_full[serie_full.index >= data_ini]

        serie = serie_full.dropna()

        if len(serie) < 20:
            logger.debug(f"{ticker}: menos de 20 observações desde a primeira compra, pulando.")
            continue

        # Dias na carteira para anualizar retorno individual
        dias_ativo = (serie.index[-1] - serie.index[0]).days
        if dias_ativo == 0:
            continue

        # Alinha IBOV com o mesmo período do ativo
        ibov_periodo   = ibov_close.reindex(serie_full.index).ffill()
        retorno_ibov   = ibov_periodo.pct_change()

        ibov_validos   = ibov_periodo.dropna()
        if len(ibov_validos) < 2:
            continue

        ret_ibov_acum  = ibov_validos.iloc[-1] / ibov_validos.iloc[0] - 1
        ret_ibov_anual = (1 + ret_ibov_acum) ** (365 / dias_ativo) - 1
        premio_ibov    = ret_ibov_anual - rfr_anual

        retorno_ativo = serie_full.pct_change()

        df_par = pd.DataFrame({
            "ativo": retorno_ativo,
            "ibov":  retorno_ibov,
        }).dropna()

        n_obs = len(df_par)
        if n_obs < 20:
            logger.debug(f"{ticker}: menos de 20 observações comuns com IBOV, pulando.")
            continue

        # Retorno acumulado e anualizado do ativo no seu próprio período
        ret_acum  = serie.iloc[-1] / serie.iloc[0] - 1
        ret_anual = (1 + ret_acum) ** (365 / dias_ativo) - 1

        # Beta = Cov(Rativo, Ribov) / Var(Ribov)
        var_ibov = df_par["ibov"].var()
        if var_ibov == 0:
            logger.debug(f"{ticker}: variância do IBOV é zero, beta não calculável.")
            continue

        beta = df_par.cov().loc["ativo", "ibov"] / var_ibov

        # Jensen alpha = Rc - [Rf + Beta * (Rm - Rf)]
        alpha_capm = ret_anual - (rfr_anual + beta * premio_ibov)

        # Treynor = (Rc - Rf) / Beta
        treynor = (ret_anual - rfr_anual) / beta if beta != 0 else np.nan

        # Volatilidade e Sharpe individuais
        vol    = df_par["ativo"].std() * np.sqrt(252)
        sharpe = (
            (df_par["ativo"].mean() - rf_diario) / df_par["ativo"].std() * np.sqrt(252)
            if df_par["ativo"].std() != 0 else np.nan
        )

        resultados.append({
            "ticker":                  ticker,
            "data_primeira_compra":    data_ini.date() if data_ini else None,
            "dias_na_carteira":        dias_ativo,
            "retorno_acumulado":       ret_acum,
            "retorno_anualizado":      ret_anual,
            "beta_ibov":               beta,
            "alpha_capm_anualizado":   alpha_capm,
            "treynor":                 treynor,
            "volatilidade_anualizada": vol,
            "sharpe_individual":       sharpe,
            "n_obs":                   n_obs,
        })

    if not resultados:
        logger.warning("Alpha por ativo CAPM: nenhum ticker com observações suficientes.")
        return pd.DataFrame()

    df_result = pd.DataFrame(resultados).sort_values("alpha_capm_anualizado", ascending=False)
    logger.info(f"Alpha CAPM por ativo calculado para {len(df_result)} ticker(s).")
    return df_result

# =========================================================
# CONCENTRAÇÃO DE RISCO POR ATIVO
# =========================================================

def calcular_concentracao_risco(
    df_ativos: pd.DataFrame,
    precos_hist: pd.DataFrame,
    df_movimentacoes: pd.DataFrame | None = None,
    min_obs: int = 21,
) -> pd.DataFrame:
    """
    Calcula a contribuição de risco de cada ativo ao risco total da carteira.

    Metodologia: Marginal Risk Contribution
        Contribuição_i = Peso_i × Cov(ativo_i, carteira) / Vol_carteira

    O histórico de cada ativo começa na sua data de primeira compra.
    A variância é calculada no período individual de cada ativo.
    A covariância entre pares usa a interseção de datas dos dois ativos.

    Parâmetros
    ----------
    df_ativos        : DataFrame com colunas 'ticker', 'valor_atual', 'quantidade_total'.
    precos_hist      : DataFrame com colunas = tickers e índice = datas.
    df_movimentacoes : DataFrame com colunas 'ticker', 'data', 'tipo'.
                       Usado para determinar a data da primeira compra de cada ativo.
    min_obs          : Mínimo de pregões desde a primeira compra. Padrão 21 (~1 mês).

    Retorna
    -------
    DataFrame ordenado por contrib_risco_pct desc.
    Ativos excluídos aparecem ao final com contrib = NaN e incluido_no_calculo = False.
    """

    if df_ativos.empty or precos_hist.empty:
        logger.warning("Concentração de risco: dados insuficientes.")
        return pd.DataFrame()

    ativos_abertos = df_ativos[
        (df_ativos["quantidade_total"] > 0) &
        (df_ativos["ticker"].isin(precos_hist.columns))
    ].copy()

    if ativos_abertos.empty:
        logger.warning("Concentração de risco: nenhum ativo aberto com histórico disponível.")
        return pd.DataFrame()

    tickers_todos = ativos_abertos["ticker"].tolist()

    # Data da primeira compra por ticker
    data_primeira_compra = {}
    if df_movimentacoes is not None and not df_movimentacoes.empty:
        compras = df_movimentacoes[df_movimentacoes["tipo"] == "COMPRA"].copy()
        compras["data"] = pd.to_datetime(compras["data"])
        data_primeira_compra = compras.groupby("ticker")["data"].min().to_dict()

    # Retornos de cada ativo a partir da sua primeira compra
    retornos_por_ticker = {}
    n_obs_por_ticker    = {}
    for t in tickers_todos:
        if t not in precos_hist.columns:
            continue
        serie = precos_hist[t].ffill()
        if t in data_primeira_compra:
            data_ini = pd.Timestamp(data_primeira_compra[t]).normalize()
            serie = serie[serie.index >= data_ini]
        ret = serie.pct_change().dropna()
        retornos_por_ticker[t] = ret
        n_obs_por_ticker[t]    = len(ret)

    tickers_ok        = [t for t in tickers_todos if n_obs_por_ticker.get(t, 0) >= min_obs]
    tickers_excluidos = [t for t in tickers_todos if t not in tickers_ok]

    for t in tickers_excluidos:
        logger.warning(
            f"Concentração de risco: {t} excluído — apenas "
            f"{n_obs_por_ticker.get(t, 0)} pregões desde a primeira compra "
            f"(mínimo: {min_obs})."
        )

    if not tickers_ok:
        logger.warning("Concentração de risco: nenhum ativo com histórico suficiente.")
        return pd.DataFrame()

    # Matriz de covariância anualizada — construída par a par
    # Cada par usa a interseção de datas; diagonal usa período completo do ativo
    n      = len(tickers_ok)
    cov_arr = np.full((n, n), 0.0)

    for i, ti in enumerate(tickers_ok):
        for j, tj in enumerate(tickers_ok):
            ri  = retornos_por_ticker[ti]
            rj  = retornos_por_ticker[tj]
            idx = ri.index.intersection(rj.index)
            if len(idx) < 5:
                continue
            if i == j:
                cov_arr[i, j] = ri.loc[idx].var() * 252
            else:
                cov_arr[i, j] = ri.loc[idx].cov(rj.loc[idx]) * 252

    cov_matrix = pd.DataFrame(cov_arr, index=tickers_ok, columns=tickers_ok)

    vol_individual = np.array([
        np.sqrt(retornos_por_ticker[t].var() * 252) for t in tickers_ok
    ])

    valores_ok = (
        ativos_abertos.set_index("ticker")["valor_atual"]
        .reindex(tickers_ok).fillna(0.0)
    )
    total = valores_ok.sum()
    if total == 0:
        return pd.DataFrame()

    pesos = (valores_ok / total).values

    var_carteira = float(pesos @ cov_matrix.values @ pesos)
    vol_carteira = np.sqrt(var_carteira) if var_carteira > 0 else np.nan

    if np.isnan(vol_carteira) or vol_carteira == 0:
        logger.warning("Concentração de risco: volatilidade da carteira inválida.")
        return pd.DataFrame()

    cov_com_carteira  = cov_matrix.values @ pesos
    contrib_risco_abs = pesos * cov_com_carteira / vol_carteira
    contrib_risco_pct = contrib_risco_abs / vol_carteira

    cov_diaria       = cov_matrix.values / 252
    var_cart_diaria  = float(pesos @ cov_diaria @ pesos)
    beta_vs_carteira = (
        (cov_diaria @ pesos) / var_cart_diaria
        if var_cart_diaria > 0 else np.full(n, np.nan)
    )

    df_ok = pd.DataFrame({
        "ticker":              tickers_ok,
        "valor_atual":         valores_ok.values,
        "peso_capital":        pesos,
        "vol_individual":      vol_individual,
        "contrib_risco_abs":   contrib_risco_abs,
        "contrib_risco_pct":   contrib_risco_pct,
        "beta_vs_carteira":    beta_vs_carteira,
        "incluido_no_calculo": True,
        "obs_historico":       [n_obs_por_ticker.get(t, 0) for t in tickers_ok],
    })

    if tickers_excluidos:
        valores_excl = (
            ativos_abertos.set_index("ticker")["valor_atual"]
            .reindex(tickers_excluidos).fillna(0.0)
        )
        df_excl = pd.DataFrame({
            "ticker":              tickers_excluidos,
            "valor_atual":         valores_excl.values,
            "peso_capital":        (valores_excl / (total + valores_excl.sum())).values,
            "vol_individual":      np.nan,
            "contrib_risco_abs":   np.nan,
            "contrib_risco_pct":   np.nan,
            "beta_vs_carteira":    np.nan,
            "incluido_no_calculo": False,
            "obs_historico":       [n_obs_por_ticker.get(t, 0) for t in tickers_excluidos],
        })
        df_resultado = pd.concat([df_ok, df_excl], ignore_index=True)
    else:
        df_resultado = df_ok

    df_resultado = df_resultado.sort_values(
        ["incluido_no_calculo", "contrib_risco_pct"],
        ascending=[False, False],
    ).reset_index(drop=True)

    logger.info(
        f"Concentração de risco — Vol carteira: {vol_carteira:.2%} | "
        f"Incluídos: {len(tickers_ok)} | Excluídos: {len(tickers_excluidos)} | "
        f"Maior: {df_ok.sort_values('contrib_risco_pct', ascending=False)['ticker'].iloc[0]} "
        f"({df_ok['contrib_risco_pct'].max():.1%})"
    )

    return df_resultado
