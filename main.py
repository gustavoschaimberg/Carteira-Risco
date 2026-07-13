import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from carteira.analytics import calcular_alpha_por_ativo_capm, calcular_concentracao_risco, calcular_metricas_risco
from carteira.config import INPUT_FILE, LOGS_DIR, OUTPUT_FILE
from carteira.data_loader import carregar_movimentacoes
from carteira.historico import construir_historico_cotizado
from carteira.market_data import (
    baixar_benchmark_ibov,
    baixar_dividendos,
    baixar_historico_precos,
    baixar_precos_recentes,
)
from carteira.portfolio_engine import (
    calcular_alpha_temporal_por_ativo,
    consolidar_carteira,
    gerar_resumo,
)
from carteira.report import exportar_excel, imprimir_resultados


def configurar_logging(verbose: bool = False):
    """Configura logging para console e arquivo."""

    nivel = logging.DEBUG if verbose else logging.INFO
    log_file = LOGS_DIR / "carteira.log"

    logging.basicConfig(
        level=nivel,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )


def parse_args():
    """Define e processa os argumentos de linha de comando."""

    parser = argparse.ArgumentParser(
        description="Consolidador de carteira de ações",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=INPUT_FILE,
        help="Caminho para o arquivo de entrada (.xlsx)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=OUTPUT_FILE,
        help="Caminho para o arquivo de saída (.xlsx)",
    )

    parser.add_argument(
        "--rfr",
        type=float,
        default=None,
        help="Taxa livre de risco anual. Se omitido, usa CDI (BCB).",
    )

    parser.add_argument(
        "--rolling",
        type=int,
        default=60,
        help="Janela (dias úteis) para métricas rolling",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Ativa log em nível DEBUG",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    configurar_logging(args.verbose)
    logger = logging.getLogger(__name__)

    logger.info("=== Iniciando consolidação da carteira ===")

    # =========================
    # CARREGAMENTO
    # =========================
    df = carregar_movimentacoes(args.input)
    tickers = sorted(df["ticker"].unique().tolist())

    # =========================
    # PREÇOS ATUAIS
    # =========================
    precos_recentes = baixar_precos_recentes(tickers)
    df_movs, df_ativos = consolidar_carteira(df, precos_recentes)
    df_resumo = gerar_resumo(df_ativos)

    imprimir_resultados(df_ativos, df_resumo)

    # =========================
    # PERÍODO
    # =========================
    data_inicio = df["data"].min()
    data_fim = pd.Timestamp.today().normalize()
    dias_carteira = (data_fim - data_inicio).days

    # =========================
    # HISTÓRICO DA CARTEIRA
    # =========================
    precos_hist = baixar_historico_precos(tickers, data_inicio, data_fim)

    # Dividendos por ticker (cache em memória — já baixados na consolidação)
    dividendos_por_ticker = {t: baixar_dividendos(t) for t in tickers}

    df_historico = construir_historico_cotizado(df, precos_hist, dividendos_por_ticker)

    # =========================
    # BENCHMARK IBOV
    # =========================
    df_ibov = baixar_benchmark_ibov(data_inicio, data_fim)

    if not df_ibov.empty:
        df_historico = df_historico.merge(
            df_ibov,
            left_on="data",
            right_index=True,
            how="left",
        )
        df_historico["IBOV"] = df_historico["IBOV"].ffill()
    else:
        logger.warning("IBOV não carregado. Alpha não será calculado.")

    # =========================
    # MÉTRICAS
    # =========================
    ibov_serie = (
        df_ibov["IBOV"].squeeze()
        if not df_ibov.empty and "IBOV" in df_ibov.columns
        else None
    )

    df_metricas_risco, df_historico_risco = calcular_metricas_risco(
        df_historico,
        taxa_livre_risco_anual=args.rfr,
        janela_rolling=args.rolling,
        ibov_close=ibov_serie,
    )

    # =========================
    # CONTRIBUIÇÃO DE ALPHA TEMPORAL
    # =========================
    df_alpha = calcular_alpha_temporal_por_ativo(
        df,
        precos_hist,
        df_ibov,
    )

    # =========================
    # ALPHA E BETA POR ATIVO — CAPM
    # =========================
    rfr_usada = (
        float(df_metricas_risco["taxa_livre_risco_anual"].iloc[0])
        if not df_metricas_risco.empty
        else (args.rfr if args.rfr is not None else 0.105)
    )
    rf_diario = (1 + rfr_usada) ** (1 / 252) - 1

    df_alpha_capm = calcular_alpha_por_ativo_capm(
        precos_hist      = precos_hist,
        ibov_close       = ibov_serie,
        rf_diario        = rf_diario,
        dias_carteira    = dias_carteira,
        df_movimentacoes = df_movs,
    )

    # =========================
    # CONCENTRAÇÃO DE RISCO POR ATIVO
    # =========================
    df_concentracao = calcular_concentracao_risco(df_ativos, precos_hist, df_movimentacoes=df_movs)

    # =========================
    # EXPORTAÇÃO
    # =========================
    exportar_excel(
        args.output,
        df_resumo,
        df_movs,
        df_ativos,
        df_historico,
        df_metricas_risco,
        df_historico_risco,
        df_alpha,
        df_alpha_capm,
        df_concentracao,
        data_referencia=data_fim,
    )

    logger.info(f"✅ Arquivo gerado com sucesso em: {args.output}")
    logger.info("=== Execução finalizada ===")


if __name__ == "__main__":
    main()