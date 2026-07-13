# Carteira de Ações

Consolidador modular de carteira de ações com métricas de risco.

## Estrutura

```
Carteira/
├── carteira/                   ← pacote Python
│   ├── __init__.py
│   ├── config.py               ← caminhos centralizados
│   ├── data_loader.py          ← leitura e validação da planilha
│   ├── market_data.py          ← downloads via yfinance
│   ├── portfolio_engine.py     ← posição, custo médio, P&L e dividendos
│   ├── historico.py            ← curva cotizada da carteira
│   ├── analytics.py            ← métricas de risco
│   ├── report.py               ← prints e exportação Excel
│   └── utils.py                ← funções utilitárias puras
│
├── tests/                      ← testes automatizados (pytest)
│   ├── test_utils.py
│   ├── test_portfolio_engine.py
│   └── test_analytics.py
│
├── data/                       ← arquivos de dados (não versionar)
│   ├── input_carteira.xlsx
│   └── output_carteira.xlsx
│
├── logs/                       ← logs gerados automaticamente
│   └── carteira.log
│
├── main.py                     ← ponto de entrada
├── requirements.txt
└── README.md
```

## Instalação

```bash
pip install -r requirements.txt
```

## Como usar

### Execução básica

Coloque `input_carteira.xlsx` dentro da pasta `data/` e rode:

```bash
python main.py
```

### Opções da linha de comando

```bash
python main.py --help
```

| Argumento | Padrão | Descrição |
|---|---|---|
| `--input` / `-i` | `data/input_carteira.xlsx` | Arquivo de entrada |
| `--output` / `-o` | `data/output_carteira.xlsx` | Arquivo de saída |
| `--rfr` | `0.0` | Taxa livre de risco anual (ex: `0.1075` para 10,75%) |
| `--rolling` | `60` | Janela em dias úteis para volatilidade rolling |
| `--verbose` / `-v` | desativado | Ativa logs em nível DEBUG |

**Exemplos:**

```bash
# Com taxa Selic de 10,75% e arquivo customizado
python main.py --rfr 0.1075 --input minha_carteira.xlsx

# Log detalhado
python main.py --verbose
```

## Testes

```bash
pytest tests/
```

## Abas geradas no Excel

| Aba | Conteúdo |
|---|---|
| Resumo | Totais consolidados da carteira |
| Movimentacoes | Histórico de compras e vendas com posição acumulada |
| Ativos | Resultado por ativo (P&L, dividendos, retorno) |
| Historico_Carteira | Valor de mercado e cota diários |
| Metricas_Risco | Sharpe, Sortino, VaR, drawdown e outros |
| Historico_Risco | Retorno diário, drawdown e volatilidade rolling |
| Grafico_Carteira | Gráfico de evolução da cota |

## Logs

Os logs são gravados simultaneamente no terminal e em `logs/carteira.log`.
Use `--verbose` para ver detalhes de cada operação.
