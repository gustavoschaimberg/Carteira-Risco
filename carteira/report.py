import logging

import pandas as pd

logger = logging.getLogger(__name__)


def imprimir_resultados(df_ativos, df_resumo):
    """Imprime resultados no terminal."""

    for _, linha in df_ativos.iterrows():
        print(f"\nAtivo: {linha['ticker']}")
        print(f"Quantidade atual: {linha['quantidade_total']:.2f}")
        print(f"Preço médio da posição aberta: R$ {linha['preco_medio_compra']:.2f}")
        print(f"Preço atual: R$ {linha['preco_atual']:.2f}")
        print(f"Valor investido em aberto: R$ {linha['valor_investido_aberto']:.2f}")
        print(f"Valor atual: R$ {linha['valor_atual']:.2f}")
        print(f"Dividendos recebidos: R$ {linha['dividendos_recebidos']:.2f}")
        print(f"Lucro realizado: R$ {linha['lucro_realizado']:.2f}")
        print(f"P&L não realizado: R$ {linha['pnl_nao_realizado']:.2f}")
        print(f"P&L total com dividendos: R$ {linha['pnl_total_com_div']:.2f}")
        print(f"Retorno % da posição aberta sem dividendos: {linha['retorno_pct_sem_div']*100:.2f}%")
        print(f"Retorno % total com dividendos: {linha['retorno_pct_com_div']*100:.2f}%")

    if not df_resumo.empty:
        resumo = df_resumo.iloc[0]
        print("\n" + "=" * 60)
        print("RESUMO DA CARTEIRA")
        print("=" * 60)
        print(f"Total investido em aberto: R$ {resumo['total_investido_aberto']:.2f}")
        print(f"Total atual: R$ {resumo['total_atual']:.2f}")
        print(f"Total de dividendos recebidos: R$ {resumo['total_dividendos_recebidos']:.2f}")
        print(f"Total de lucro realizado: R$ {resumo['total_lucro_realizado']:.2f}")
        print(f"Total de P&L não realizado: R$ {resumo['total_pnl_nao_realizado']:.2f}")
        print(f"Total de P&L com dividendos: R$ {resumo['total_pnl_com_dividendos']:.2f}")
        print(f"Retorno total: {resumo['retorno_total_pct']*100:.2f}%")


def exportar_excel(
    output_path,
    df_resumo,
    df_movs,
    df_ativos,
    df_historico,
    df_metricas_risco=None,
    df_historico_risco=None,
    df_alpha=None,
    df_alpha_capm=None,
    df_concentracao=None,
    data_referencia=None,
):
    """Exporta os resultados para Excel com formatação e gráfico."""

    logger.info(f"Exportando resultado para: {output_path}")

    try:
        with pd.ExcelWriter(
            output_path,
            engine="xlsxwriter",
            datetime_format="yyyy-mm-dd",
        ) as writer:

            # Injeta data de referência no Resumo antes de exportar
            df_resumo_export = df_resumo.copy()
            if data_referencia is not None:
                df_resumo_export.insert(0, "data_referencia", pd.Timestamp(data_referencia).date())

            df_resumo_export.to_excel(writer, sheet_name="Resumo", index=False)
            df_movs.to_excel(writer, sheet_name="Movimentacoes", index=False)

            df_ativos_export = df_ativos.drop(columns=["capital_investido_total"], errors="ignore")
            df_ativos_export.to_excel(writer, sheet_name="Ativos", index=False)

            df_historico.to_excel(writer, sheet_name="Historico_Carteira", index=False)

            if df_metricas_risco is not None and not df_metricas_risco.empty:
                df_metricas_risco.to_excel(writer, sheet_name="Metricas_Risco", index=False)

            if df_historico_risco is not None and not df_historico_risco.empty:
                df_historico_risco.to_excel(writer, sheet_name="Historico_Risco", index=False)

            if df_alpha is not None and not df_alpha.empty:
                df_alpha.to_excel(writer, sheet_name="Alpha_Ativos", index=False)

            if df_alpha_capm is not None and not df_alpha_capm.empty:
                df_alpha_capm.to_excel(writer, sheet_name="Alpha_CAPM_Ativos", index=False)

            if df_concentracao is not None and not df_concentracao.empty:
                df_concentracao.to_excel(writer, sheet_name="Concentracao_Risco", index=False)

            workbook = writer.book
            fmt_money = workbook.add_format({"num_format": "R$ #,##0.00"})
            fmt_pct = workbook.add_format({"num_format": "0.00%"})
            fmt_num = workbook.add_format({"num_format": "0.0000"})
            fmt_date = workbook.add_format({"num_format": "yyyy-mm-dd"})
            fmt_header = workbook.add_format({"bold": True, "bg_color": "#D9EAF7", "border": 1})

            abas = {
                "Resumo": df_resumo_export,
                "Movimentacoes": df_movs,
                "Ativos": df_ativos_export,
                "Historico_Carteira": df_historico,
                "Metricas_Risco": df_metricas_risco,
                "Historico_Risco": df_historico_risco,
                "Alpha_Ativos": df_alpha,
                "Alpha_CAPM_Ativos": df_alpha_capm,
                "Concentracao_Risco": df_concentracao,
            }

            for sheet_name, frame in abas.items():
                if sheet_name in writer.sheets and frame is not None and not frame.empty:
                    ws = writer.sheets[sheet_name]
                    for col_num, value in enumerate(frame.columns.values):
                        ws.write(0, col_num, value, fmt_header)

            ws_resumo = writer.sheets["Resumo"]
            ws_movs   = writer.sheets["Movimentacoes"]
            ws_ativos = writer.sheets["Ativos"]
            ws_hist   = writer.sheets["Historico_Carteira"]

            ws_resumo.set_column("A:A", 18, fmt_date)   # data_referencia
            ws_resumo.set_column("B:G", 24, fmt_money)
            ws_resumo.set_column("H:H", 18)
            ws_resumo.set_column("I:I", 18, fmt_pct)

            ws_movs.set_column("A:B", 14)
            ws_movs.set_column("C:C", 12)
            ws_movs.set_column("D:D", 12)
            ws_movs.set_column("E:I", 18, fmt_money)
            ws_movs.set_column("J:J", 18, fmt_money)

            ws_ativos.set_column("A:A", 14)
            ws_ativos.set_column("B:B", 14)
            ws_ativos.set_column("C:J", 18, fmt_money)
            ws_ativos.set_column("K:L", 18, fmt_pct)

            ws_hist.set_column("A:A", 14, fmt_date)
            ws_hist.set_column("B:C", 18, fmt_money)
            ws_hist.set_column("D:E", 18, fmt_num)    # cotas, cota
            ws_hist.set_column("F:F", 18, fmt_money)  # caixa_dividendos

            if "Metricas_Risco" in writer.sheets:
                ws_risco = writer.sheets["Metricas_Risco"]

                ws_risco.set_column("A:A", 24, fmt_pct)
                ws_risco.set_column("B:B", 28)
                ws_risco.set_column("C:G", 24, fmt_pct)
                ws_risco.set_column("H:I", 18, fmt_num)
                ws_risco.set_column("J:L", 20, fmt_pct)
                ws_risco.set_column("M:O", 24, fmt_pct)

            if "Historico_Risco" in writer.sheets:
                ws_hr = writer.sheets["Historico_Risco"]
                ws_hr.set_column("A:A", 14, fmt_date)
                ws_hr.set_column("B:H", 18)

            if "Alpha_Ativos" in writer.sheets:
                ws_alpha = writer.sheets["Alpha_Ativos"]
                ws_alpha.set_column("A:A", 16)
                ws_alpha.set_column("B:E", 22, fmt_pct)

            if "Alpha_CAPM_Ativos" in writer.sheets:
                ws_capm = writer.sheets["Alpha_CAPM_Ativos"]
                fmt_int = workbook.add_format({"num_format": "0"})
                ws_capm.set_column("A:A", 16)
                ws_capm.set_column("B:C", 22, fmt_pct)
                ws_capm.set_column("D:D", 14, fmt_num)
                ws_capm.set_column("E:G", 22, fmt_pct)
                ws_capm.set_column("H:H", 22, fmt_num)
                ws_capm.set_column("I:I", 12, fmt_int)

            if "Concentracao_Risco" in writer.sheets:
                ws_conc = writer.sheets["Concentracao_Risco"]
                fmt_int = workbook.add_format({"num_format": "0"})
                ws_conc.set_column("A:A", 14)            # ticker
                ws_conc.set_column("B:B", 20, fmt_money) # valor_atual
                ws_conc.set_column("C:C", 18, fmt_pct)   # peso_capital
                ws_conc.set_column("D:D", 22, fmt_pct)   # vol_individual
                ws_conc.set_column("E:E", 22, fmt_pct)   # contrib_risco_abs
                ws_conc.set_column("F:F", 22, fmt_pct)   # contrib_risco_pct
                ws_conc.set_column("G:G", 20, fmt_num)   # beta_vs_carteira
                ws_conc.set_column("H:H", 22)            # incluido_no_calculo
                ws_conc.set_column("I:I", 16, fmt_int)   # obs_historico

            ws_graf = workbook.add_worksheet("Grafico_Carteira")

            if not df_historico.empty:
                chart = workbook.add_chart({"type": "line"})
                max_row = len(df_historico)

                chart.add_series({
                    "name": "Valor da Cota",
                    "categories": f"=Historico_Carteira!$A$2:$A${max_row + 1}",
                    "values": f"=Historico_Carteira!$E$2:$E${max_row + 1}",
                    "marker": {"type": "none"},
                })

                chart.set_title({"name": "Valor da Cota"})
                chart.set_x_axis({"name": "Data", "date_axis": True})
                chart.set_y_axis({"name": "Cota", "num_format": "0.0000"})
                chart.set_legend({"none": True})
                chart.set_size({"width": 900, "height": 500})

                ws_graf.insert_chart("B2", chart)
            else:
                ws_graf.write("B2", "Não foi possível gerar o histórico da carteira.")

        logger.info("Exportação concluída com sucesso.")

    except PermissionError:
        logger.error(f"Arquivo '{output_path}' está aberto no Excel. Feche-o e tente novamente.")
        raise