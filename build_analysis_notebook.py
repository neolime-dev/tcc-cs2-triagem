#!/usr/bin/env python3
"""Constrói o notebook principal de análises do TCC."""
from pathlib import Path
import nbformat as nbf

HERE = Path(__file__).parent
OUT = HERE / "ANALISES_TCC_CS2.ipynb"
nb = nbf.v4.new_notebook()
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3 (TCC CS2)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.14"},
}

cells = []
md = lambda text: cells.append(nbf.v4.new_markdown_cell(text.strip()))
code = lambda text: cells.append(nbf.v4.new_code_cell(text.strip()))

md(r"""
# Análises exploratórias de preços e volumes de skins de CS2

**Autor:** Matheus Lima Ribeiro  
**Projeto:** TCC — Ciência de Dados e Inteligência Artificial — IESB  
**Próxima orientação:** 09/09/2026, 8h45

## Pergunta provisória

Como identificar padrões anômalos de preço e volume em itens de CS2 no Steam Community Market e avaliar se esses padrões podem constituir sinais para investigação de possível manipulação?

## Regra de interpretação

Este notebook identifica **comportamentos estatísticos incomuns**. Ele não confirma fraude ou manipulação. O dataset é publicado por terceiro e ainda precisa de validação externa.
""")

md(r"""
## Como explicar qualquer gráfico

Para cada visualização, responda:

1. **Pergunta:** o que o gráfico tenta investigar?
2. **Dados:** quais linhas e colunas foram usadas?
3. **Transformação:** houve log, retorno, normalização ou janela móvel?
4. **Eixos:** o que representam X, Y, cores e marcas?
5. **Observação:** qual padrão está visível?
6. **Limite:** o que o gráfico não permite concluir?
7. **Próximo teste:** o que deveria ser verificado depois?
""")

md("## 1. Ambiente, caminhos e reprodutibilidade")
code(r"""
from pathlib import Path
import gc
import json
import warnings

import matplotlib
matplotlib.use("module://matplotlib_inline.backend_inline")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.compute as pc
import ruptures as rpt
import seaborn as sns
from IPython.display import display, Markdown
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.figsize": (11, 5.5), "axes.titlesize": 13, "axes.labelsize": 11})

HERE = Path.cwd()
META = json.loads((HERE / "data_source.json").read_text(encoding="utf-8"))
DATA_DIR = Path(META["download_path"])
PARQUET = DATA_DIR / "daily_prices.parquet"
ITEMS_CSV = DATA_DIR / "items.csv"
FIG_DIR = HERE / "notebook_figures"
FIG_DIR.mkdir(exist_ok=True)

assert PARQUET.exists(), "Execute download_data.py antes do notebook."
assert ITEMS_CSV.exists(), "Arquivo items.csv não encontrado."
print("Dataset:", META["handle"])
print("Licença declarada:", META["license_declared"])
print("Parquet:", PARQUET)
""")

md(r"""
### Fonte e cautela

O dataset usado é `devynpruden/cs2-skin-price-history-2013-2026`, versão 4, com licença declarada Apache 2.0. A descrição informa origem em dados do mercado Steam. Essa origem ainda não foi validada de forma independente.

**Como explicar:** a licença e a ausência de valores faltantes não provam que os preços estejam corretos. Confiabilidade externa exige conferir amostras e entender o processo de coleta.
""")

md("## 2. Auditoria estrutural do conjunto completo")
code(r"""
audit = json.loads((HERE / "auditoria_dados.json").read_text(encoding="utf-8"))
audit_summary = pd.DataFrame({
    "métrica": ["linhas", "itens distintos", "data inicial", "data final", "duplicatas item-data", "linhas do catálogo"],
    "valor": [audit["rows"], audit["distinct_items"], audit["date_min"], audit["date_max"], audit["duplicate_item_date_rows"], audit["items_catalog_rows"]],
})
display(audit_summary)
missing = pd.Series(audit["missing"], name="ausências").sort_values(ascending=False)
display(missing.to_frame().head(20))
""")

md(r"""
**Como explicar:** esta tabela responde se o arquivo tem estrutura suficiente para análise. Encontramos mais de 10 milhões de linhas, 4.342 séries, nenhuma duplicata item-data e nenhuma ausência registrada. Isso comprova consistência interna, não fidelidade à Steam.
""")

md("## 3. Catálogo: composição dos itens")
code(r"""
items = pd.read_csv(ITEMS_CSV, low_memory=False)
print("Linhas do catálogo:", len(items))
print("Itens únicos:", items["market_hash_name"].nunique())
display(items.head(3))
daily_source_counts = pc.value_counts(pq.read_table(PARQUET, columns=["source"])["source"]).to_pylist()
print("Origem declarada no histórico:", daily_source_counts)
print("Origem declarada no catálogo:", items["source"].value_counts(dropna=False).to_dict())

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
items["weapon"].value_counts().head(15).sort_values().plot.barh(ax=axes[0], color="#4472C4")
axes[0].set_title("15 armas com mais variantes no catálogo")
axes[0].set_xlabel("Quantidade de variantes")
axes[0].set_ylabel("Arma")
items["exterior"].value_counts().plot.bar(ax=axes[1], color="#70AD47")
axes[1].set_title("Variantes por condição de desgaste")
axes[1].set_xlabel("Condição")
axes[1].set_ylabel("Quantidade")
axes[1].tick_params(axis="x", rotation=35)
plt.tight_layout()
plt.savefig(FIG_DIR / "01_catalogo_composicao.png", dpi=160, bbox_inches="tight")
plt.show()
""")

md(r"""
**Como explicar:** as barras contam variantes no catálogo, não vendas. Elas mostram como o universo de itens é composto e ajudam a evitar uma amostra concentrada em uma única arma ou condição. Não dizem quais itens são mais populares.

**Proveniência observada:** todas as linhas do histórico têm `source = steam_ssr`, enquanto o catálogo tem `source = skinport` e URLs da Skinport. O dataset combina componentes de origens declaradas diferentes. Isso exige validação adicional antes de tratá-lo como fonte final.
""")

md("## 4. Cobertura histórica por item")
code(r"""
minimal = pd.read_parquet(PARQUET, columns=["market_hash_name", "date", "price_median", "volume"])
minimal["date"] = pd.to_datetime(minimal["date"])
coverage = minimal.groupby("market_hash_name").agg(
    observacoes=("date", "size"),
    inicio=("date", "min"),
    fim=("date", "max"),
    volume_total=("volume", "sum"),
    preco_mediano=("price_median", "median"),
).reset_index()
coverage["anos_cobertos"] = (coverage["fim"] - coverage["inicio"]).dt.days / 365.25
coverage["ano_inicio"] = coverage["inicio"].dt.year

display(coverage.describe(include="all").T)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.histplot(coverage["observacoes"], bins=50, ax=axes[0], color="#4472C4")
axes[0].set_title("Distribuição do número de dias por item")
axes[0].set_xlabel("Observações diárias")
axes[0].set_ylabel("Quantidade de itens")
coverage["ano_inicio"].value_counts().sort_index().plot.bar(ax=axes[1], color="#ED7D31")
axes[1].set_title("Ano de início das séries")
axes[1].set_xlabel("Ano")
axes[1].set_ylabel("Quantidade de itens")
plt.tight_layout()
plt.savefig(FIG_DIR / "02_cobertura_historica.png", dpi=160, bbox_inches="tight")
plt.show()
""")

md(r"""
**Como explicar:** o histograma mostra que os itens têm quantidades diferentes de dias registrados. O segundo gráfico mostra quando as séries começam. Isso impede assumir que todas as skins podem ser comparadas no mesmo período. Cobertura longa também não garante coleta correta.
""")

code(r"""
# Libera o conjunto de mais de 10 milhões de linhas antes da análise detalhada.
del minimal
gc.collect()
""")

md("## 5. Cesta piloto e preparação das séries")
code(r"""
SELECTED_ITEMS = [
    "AK-47 | Asiimov (Field-Tested)",
    "AWP | Hyper Beast (Field-Tested)",
    "M4A1-S | Printstream (Field-Tested)",
    "Desert Eagle | Printstream (Field-Tested)",
    "Glock-18 | Water Elemental (Field-Tested)",
]
START = pd.Timestamp("2024-01-01")
cols = ["market_hash_name", "date", "price_median", "price_mean", "price_min", "price_max", "volume", "source_point_count", "weapon", "exterior"]
df = pd.read_parquet(
    PARQUET,
    columns=cols,
    filters=[("market_hash_name", "in", SELECTED_ITEMS), ("date", ">=", START.date())],
)
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["market_hash_name", "date"]).reset_index(drop=True)
df["short_name"] = df["market_hash_name"].str.replace(" (Field-Tested)", "", regex=False)
df["gap_days"] = df.groupby("market_hash_name")["date"].diff().dt.days
df["log_price"] = np.log(df["price_median"])
df["log_return"] = df.groupby("market_hash_name")["log_price"].diff()
df.loc[df["gap_days"] > 3, "log_return"] = np.nan
df["log_volume"] = np.log1p(df["volume"])
df["log_volume_change"] = df.groupby("market_hash_name")["log_volume"].diff()
df.loc[df["gap_days"] > 3, "log_volume_change"] = np.nan
df["range_pct"] = (df["price_max"] - df["price_min"]) / df["price_median"]
df["rolling_vol_30"] = df.groupby("market_hash_name")["log_return"].transform(lambda s: s.rolling(30, min_periods=15).std())
df["rolling_volume_30"] = df.groupby("market_hash_name")["volume"].transform(lambda s: s.rolling(30, min_periods=15).median())
df["volume_ratio_30"] = df["volume"] / df["rolling_volume_30"]
print("Linhas da cesta:", len(df))
print("Período:", df["date"].min().date(), "a", df["date"].max().date())
print("Maior lacuna:", df["gap_days"].max(), "dias")
""")

code(r"""
summary = df.groupby("short_name").agg(
    observacoes=("date", "size"),
    preco_mediano=("price_median", "median"),
    preco_minimo=("price_median", "min"),
    preco_maximo=("price_median", "max"),
    volume_total=("volume", "sum"),
    volume_mediano=("volume", "median"),
    maior_lacuna=("gap_days", "max"),
).sort_values("volume_total", ascending=False)
display(summary)
""")

md(r"""
**Como explicar:** a cesta fixa a condição Field-Tested para reduzir uma fonte de diferença, mas ainda mistura armas, idades e níveis de liquidez. Ela é piloto, não amostra representativa de todo o mercado.
""")

md("## 6. Distribuições de preço e volume")
code(r"""
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
sns.boxplot(data=df, y="short_name", x="price_median", ax=axes[0], color="#5B9BD5")
axes[0].set_xscale("log")
axes[0].set_title("Distribuição do preço mediano diário")
axes[0].set_xlabel("Preço mediano em USD — escala log")
axes[0].set_ylabel("")
sns.boxplot(data=df, y="short_name", x="volume", ax=axes[1], color="#70AD47", showfliers=False)
axes[1].set_xscale("log")
axes[1].set_title("Distribuição do volume diário")
axes[1].set_xlabel("Volume — escala log; outliers ocultos no desenho")
axes[1].set_ylabel("")
plt.tight_layout()
plt.savefig(FIG_DIR / "03_distribuicoes_preco_volume.png", dpi=160, bbox_inches="tight")
plt.show()
""")

md(r"""
**Como explicar:** a escala logarítmica permite comparar itens com níveis muito diferentes. A caixa resume mediana e dispersão; no volume, pontos extremos foram ocultados apenas do desenho para manter a leitura. Eles continuam nos dados. O gráfico não mede rentabilidade nem causalidade.
""")

md("## 7. Evolução temporal do preço")
code(r"""
fig, axes = plt.subplots(5, 1, figsize=(13, 14), sharex=True)
for ax, (name, g) in zip(axes, df.groupby("short_name", sort=False)):
    ax.plot(g["date"], g["price_median"], linewidth=1.2, color="#4472C4")
    ax.set_title(name, loc="left", fontsize=11)
    ax.set_ylabel("USD")
axes[-1].set_xlabel("Data")
fig.suptitle("Preço mediano diário por skin", y=1.002, fontsize=14)
plt.tight_layout()
plt.savefig(FIG_DIR / "04_series_preco.png", dpi=160, bbox_inches="tight")
plt.show()
""")

md(r"""
**Como explicar:** cada painel usa sua própria série, mas compartilha o tempo. Mudanças de nível e picos ficam visíveis. Um pico mostra quando investigar; não explica o motivo.
""")

md("## 8. Preços normalizados para base 100")
code(r"""
fig, ax = plt.subplots(figsize=(12, 6))
for name, g in df.groupby("short_name"):
    g = g.sort_values("date")
    indexed = 100 * g["price_median"] / g["price_median"].iloc[0]
    ax.plot(g["date"], indexed, linewidth=1.2, label=name)
ax.axhline(100, color="black", linestyle="--", linewidth=0.8)
ax.set_yscale("log")
ax.set_title("Trajetórias relativas de preço — base 100")
ax.set_xlabel("Data")
ax.set_ylabel("Índice de preço — escala log")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIG_DIR / "05_precos_normalizados.png", dpi=160, bbox_inches="tight")
plt.show()
""")

md(r"""
**Como explicar:** todas as séries começam em 100, removendo a diferença de preço absoluto. O eixo logarítmico evita que a skin com maior valorização esconda as outras. Base 100 depende da data inicial e não representa retorno ajustado a risco.
""")

md("## 9. Distribuição dos retornos diários")
code(r"""
fig, axes = plt.subplots(3, 2, figsize=(13, 12))
axes = axes.flatten()
for ax, (name, g) in zip(axes, df.groupby("short_name")):
    sns.histplot(g["log_return"].dropna() * 100, bins=60, ax=ax, color="#4472C4", stat="density")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(name, fontsize=10)
    ax.set_xlabel("Retorno logarítmico diário (%)")
    ax.set_ylabel("Densidade")
axes[-1].axis("off")
plt.tight_layout()
plt.savefig(FIG_DIR / "06_distribuicao_retornos.png", dpi=160, bbox_inches="tight")
plt.show()

return_stats = df.groupby("short_name")["log_return"].agg(["count", "mean", "std", "skew", "min", "max"])
return_stats[["mean", "std", "min", "max"]] *= 100
return_stats = return_stats.rename(columns={"mean":"média_%", "std":"desvio_%", "min":"mínimo_%", "max":"máximo_%"})
display(return_stats)
return_stats.to_csv(HERE / "output" / "estatisticas_retornos.csv")
""")

md(r"""
**Como explicar:** retornos transformam preço em variação diária e permitem comparar níveis diferentes. Caudas longas indicam dias extremos, mas não dizem se houve erro, evento legítimo ou manipulação. Média próxima de zero não significa ausência de risco.
""")

md("## 10. Volatilidade móvel de 30 dias")
code(r"""
fig, ax = plt.subplots(figsize=(12, 6))
for name, g in df.groupby("short_name"):
    ax.plot(g["date"], g["rolling_vol_30"] * 100, label=name, linewidth=1.1)
ax.set_title("Volatilidade móvel dos retornos — janela de 30 observações")
ax.set_xlabel("Data")
ax.set_ylabel("Desvio-padrão móvel (%)")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(FIG_DIR / "07_volatilidade_movel.png", dpi=160, bbox_inches="tight")
plt.show()
""")

md(r"""
**Como explicar:** cada ponto resume a dispersão dos 30 retornos anteriores. Picos significam instabilidade recente, não direção do preço. A janela de 30 é uma escolha e precisa de sensibilidade.
""")

md("## 11. Risco, retorno e liquidez aproximada")
code(r"""
risk_return = df.groupby("short_name").agg(
    retorno_diario_medio=("log_return", "mean"),
    volatilidade_diaria=("log_return", "std"),
    volume_mediano=("volume", "median"),
).reset_index()
risk_return[["retorno_diario_medio", "volatilidade_diaria"]] *= 100

fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=risk_return, x="volatilidade_diaria", y="retorno_diario_medio", size="volume_mediano", color="#4472C4", sizes=(80, 500), legend="brief", ax=ax)
for _, row in risk_return.iterrows():
    ax.annotate(row["short_name"].split(" | ")[0], (row["volatilidade_diaria"], row["retorno_diario_medio"]), xytext=(4,4), textcoords="offset points", fontsize=8)
ax.set_title("Retorno médio diário, volatilidade e volume mediano")
ax.set_xlabel("Volatilidade diária (%)")
ax.set_ylabel("Retorno logarítmico médio diário (%)")
ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(FIG_DIR / "08_risco_retorno_volume.png", dpi=160, bbox_inches="tight")
plt.show()
display(risk_return)
""")

md(r"""
**Como explicar:** X representa instabilidade, Y representa média diária e o tamanho aproxima liquidez pelo volume mediano. São estatísticas descritivas, sem custos, sem risco ajustado e sem inferência de investimento.
""")

md("## 12. Relação contemporânea entre retorno e mudança de volume")
code(r"""
correlations = []
for name, g in df.groupby("short_name"):
    valid = g[["log_return", "log_volume_change"]].dropna()
    rho, pvalue = spearmanr(valid["log_return"], valid["log_volume_change"])
    correlations.append({"item": name, "rho_spearman": rho, "p_value": pvalue, "n": len(valid)})
corr_df = pd.DataFrame(correlations).sort_values("rho_spearman")
display(corr_df)
corr_df.to_csv(HERE / "output" / "correlacao_retorno_volume.csv", index=False)

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(data=corr_df, y="item", x="rho_spearman", color="#5B9BD5", ax=ax)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Correlação de Spearman: retorno e mudança de volume no mesmo dia")
ax.set_xlabel("Coeficiente de Spearman")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig(FIG_DIR / "09_correlacao_retorno_volume.png", dpi=160, bbox_inches="tight")
plt.show()
""")

md(r"""
**Como explicar:** Spearman mede associação monotônica no mesmo dia. Valor positivo indica que retornos maiores tendem a acompanhar aumentos de volume; negativo indica direção oposta. Correlação não prova causalidade, coordenação nem manipulação. O p-valor também não mede importância prática.
""")

md("## 13. Padrões por dia da semana e mês")
code(r"""
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
df["weekday"] = pd.Categorical(df["date"].dt.day_name(), categories=weekday_order, ordered=True)
weekday = df.groupby("weekday", observed=False).agg(
    retorno_absoluto_medio=("log_return", lambda s: s.abs().mean() * 100),
    volume_mediano=("volume", "median"),
).reset_index()
weekday["dia"] = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.barplot(data=weekday, x="dia", y="retorno_absoluto_medio", ax=axes[0], color="#ED7D31")
axes[0].set_title("Variação absoluta média por dia da semana")
axes[0].set_xlabel("")
axes[0].set_ylabel("Retorno absoluto médio (%)")
axes[0].tick_params(axis="x", rotation=35)
sns.barplot(data=weekday, x="dia", y="volume_mediano", ax=axes[1], color="#70AD47")
axes[1].set_title("Volume mediano por dia da semana")
axes[1].set_xlabel("")
axes[1].set_ylabel("Volume mediano")
axes[1].tick_params(axis="x", rotation=35)
plt.tight_layout()
plt.savefig(FIG_DIR / "10_dia_semana.png", dpi=160, bbox_inches="tight")
plt.show()

focus = df[df["market_hash_name"] == SELECTED_ITEMS[0]].copy()
focus["year"] = focus["date"].dt.year
focus["month"] = focus["date"].dt.month
heat = focus.pivot_table(index="year", columns="month", values="log_return", aggfunc="mean") * 100
fig, ax = plt.subplots(figsize=(11, 4))
sns.heatmap(heat, cmap="RdBu_r", center=0, annot=True, fmt=".2f", ax=ax)
ax.set_title("AK-47 Asiimov: retorno médio por ano e mês (%)")
ax.set_xlabel("Mês")
ax.set_ylabel("Ano")
plt.tight_layout()
plt.savefig(FIG_DIR / "11_sazonalidade_mensal.png", dpi=160, bbox_inches="tight")
plt.show()
""")

md(r"""
**Como explicar:** os gráficos agregam observações por calendário. Diferenças podem sugerir sazonalidade, mas a amostra tem poucos anos e cinco itens. Um padrão visual não prova efeito causal; eventos específicos podem dominar um mês.
""")

md("## 14. Correlação entre retornos das skins")
code(r"""
return_panel = df.pivot(index="date", columns="short_name", values="log_return")
return_corr = return_panel.corr(method="spearman")
fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(return_corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax)
ax.set_title("Correlação de Spearman entre retornos diários")
plt.tight_layout()
plt.savefig(FIG_DIR / "12_correlacao_skins.png", dpi=160, bbox_inches="tight")
plt.show()
display(return_corr)
return_corr.to_csv(HERE / "output" / "correlacao_retornos_skins.csv")
""")

md(r"""
**Como explicar:** cada célula compara a direção relativa dos retornos de duas skins. Correlação alta pode indicar exposição a fatores comuns, mas não prova que uma skin cause movimento em outra. A correlação pode mudar no tempo.
""")

md("## 15. Baseline robusto de eventos candidatos")
code(r"""
def robust_z(series):
    median = series.median(skipna=True)
    mad = (series - median).abs().median(skipna=True)
    if not np.isfinite(mad) or mad == 0:
        return pd.Series(np.nan, index=series.index)
    return (series - median) / (1.4826 * mad)

for column, output in [
    ("log_return", "z_return"),
    ("log_volume", "z_volume"),
    ("log_volume_change", "z_volume_change"),
    ("rolling_vol_30", "z_rolling_vol"),
]:
    df[output] = df.groupby("market_hash_name")[column].transform(robust_z)

THRESHOLD = 4.0
df["return_candidate"] = df["z_return"].abs() >= THRESHOLD
df["volume_candidate"] = df["z_volume"].abs() >= THRESHOLD
df["robust_candidate"] = df["return_candidate"] | df["volume_candidate"]
df["robust_score"] = df[["z_return", "z_volume"]].abs().max(axis=1)

candidate_counts = df.groupby("short_name").agg(
    observacoes=("date", "size"),
    retorno_candidato=("return_candidate", "sum"),
    volume_candidato=("volume_candidate", "sum"),
    uniao=("robust_candidate", "sum"),
)
candidate_counts["percentual_uniao"] = 100 * candidate_counts["uniao"] / candidate_counts["observacoes"]
display(candidate_counts)
candidate_counts.to_csv(HERE / "output" / "contagem_eventos_mad.csv")
""")

code(r"""
fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
f = df[df["market_hash_name"] == SELECTED_ITEMS[0]]
ret = f[f["return_candidate"]]
vol = f[f["volume_candidate"]]
axes[0].plot(f["date"], f["price_median"], color="#4472C4", linewidth=1.1)
axes[0].scatter(ret["date"], ret["price_median"], color="#C00000", s=22, label="Retorno candidato")
axes[0].set_ylabel("Preço mediano (USD)")
axes[0].legend()
axes[1].plot(f["date"], f["volume"], color="#70AD47", linewidth=1.0)
axes[1].scatter(vol["date"], vol["volume"], color="#C00000", s=22, label="Volume candidato")
axes[1].set_ylabel("Volume")
axes[1].set_xlabel("Data")
axes[1].legend()
fig.suptitle("AK-47 Asiimov: eventos candidatos pelo baseline robusto")
plt.tight_layout()
plt.savefig(FIG_DIR / "13_eventos_robustos_ak47.png", dpi=160, bbox_inches="tight")
plt.show()

top_events = df[df["robust_candidate"]].nlargest(20, "robust_score")[[
    "date", "short_name", "price_median", "volume", "log_return", "z_return", "z_volume", "robust_score"
]]
display(top_events)
""")

md(r"""
**Como explicar:** pontos vermelhos ultrapassam quatro MADs no retorno ou volume. A mediana e o MAD reduzem influência de extremos na referência. O limiar 4 é uma escolha exploratória; os pontos são candidatos, não fraudes.
""")

md("## 16. Sensibilidade do limiar robusto")
code(r"""
thresholds = [3.0, 3.5, 4.0, 4.5, 5.0, 6.0]
sensitivity = []
base_score = df[["z_return", "z_volume"]].abs().max(axis=1)
for threshold in thresholds:
    sensitivity.append({"limiar": threshold, "eventos": int((base_score >= threshold).sum()), "percentual": 100 * (base_score >= threshold).mean()})
sensitivity_df = pd.DataFrame(sensitivity)
display(sensitivity_df)
sensitivity_df.to_csv(HERE / "output" / "sensibilidade_limiar_mad.csv", index=False)
fig, ax = plt.subplots(figsize=(8, 5))
sns.lineplot(data=sensitivity_df, x="limiar", y="eventos", marker="o", ax=ax)
ax.set_title("Sensibilidade: número de eventos por limiar robusto")
ax.set_xlabel("Limiar absoluto")
ax.set_ylabel("Eventos candidatos")
plt.tight_layout()
plt.savefig(FIG_DIR / "14_sensibilidade_limiar.png", dpi=160, bbox_inches="tight")
plt.show()
""")

md(r"""
**Como explicar:** aumentar o limiar reduz eventos. Se a conclusão muda completamente com pequena alteração, o método é instável. O gráfico ajuda a escolher um limiar transparente, mas não revela qual limiar corresponde a manipulação real.
""")

md("## 17. PELT: pontos de mudança e sensibilidade")
code(r"""
f = df[df["market_hash_name"] == SELECTED_ITEMS[0]].copy().reset_index(drop=True)
signal = np.column_stack([
    (f["log_price"] - f["log_price"].mean()) / f["log_price"].std(),
    (f["log_volume"] - f["log_volume"].mean()) / f["log_volume"].std(),
])
penalties = [5, 8, 10, 12, 15, 20]
pelt_rows = []
pelt_breaks = {}
for penalty in penalties:
    breaks = rpt.Pelt(model="rbf", min_size=14, jump=1).fit(signal).predict(pen=penalty)
    breaks = [b for b in breaks if b < len(f)]
    pelt_breaks[penalty] = breaks
    pelt_rows.append({"penalizacao": penalty, "pontos_mudanca": len(breaks)})
pelt_sensitivity = pd.DataFrame(pelt_rows)
display(pelt_sensitivity)
pelt_sensitivity.to_csv(HERE / "output" / "sensibilidade_pelt.csv", index=False)
pd.DataFrame(
    [{"penalizacao": p, "ordem": i + 1, "indice": b, "data": f.iloc[b]["date"].date()} for p, bs in pelt_breaks.items() for i, b in enumerate(bs)]
).to_csv(HERE / "output" / "pontos_mudanca_pelt_notebook.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.lineplot(data=pelt_sensitivity, x="penalizacao", y="pontos_mudanca", marker="o", ax=axes[0])
axes[0].set_title("Sensibilidade do PELT à penalização")
axes[0].set_xlabel("Penalização")
axes[0].set_ylabel("Quantidade de mudanças")
axes[1].plot(f["date"], f["price_median"], color="#4472C4", linewidth=1.1)
for i, b in enumerate(pelt_breaks[10]):
    axes[1].axvline(f.iloc[b]["date"], color="#C55A11", alpha=0.65, linewidth=1, label="Mudança PELT" if i == 0 else None)
axes[1].set_title("AK-47 Asiimov: mudanças com penalização 10")
axes[1].set_xlabel("Data")
axes[1].set_ylabel("Preço mediano (USD)")
axes[1].legend()
locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
axes[1].xaxis.set_major_locator(locator)
axes[1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
plt.tight_layout()
plt.savefig(FIG_DIR / "15_pelt_sensibilidade.png", dpi=160, bbox_inches="tight")
plt.show()
""")

md(r"""
**Como explicar:** PELT divide a série em segmentos. Penalização baixa aceita mais mudanças; penalização alta exige evidência maior. Linhas verticais indicam transições estatísticas, não manipulação. O parâmetro 10 é demonstrativo.
""")

md("## 18. Isolation Forest como comparador multivariado")
code(r"""
FEATURES = ["z_return", "z_volume", "z_volume_change", "z_rolling_vol"]
model_data = df.dropna(subset=FEATURES).copy()
iso = IsolationForest(n_estimators=300, contamination=0.02, random_state=42, n_jobs=-1)
model_data["if_label"] = iso.fit_predict(model_data[FEATURES])
model_data["if_score"] = -iso.decision_function(model_data[FEATURES])
model_data["if_candidate"] = model_data["if_label"] == -1

if_summary = model_data.groupby("short_name").agg(
    observacoes=("date", "size"),
    eventos_if=("if_candidate", "sum"),
    eventos_robustos=("robust_candidate", "sum"),
).reset_index()
if_summary["sobreposicao"] = model_data.groupby("short_name").apply(lambda g: int((g["if_candidate"] & g["robust_candidate"]).sum()), include_groups=False).values
display(if_summary)
if_summary.to_csv(HERE / "output" / "resumo_isolation_forest.csv", index=False)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_data = model_data[model_data["market_hash_name"] == SELECTED_ITEMS[0]]
axes[0].plot(plot_data["date"], plot_data["if_score"], color="#7030A0", linewidth=1.0)
marked = plot_data[plot_data["if_candidate"]]
axes[0].scatter(marked["date"], marked["if_score"], color="#C00000", s=20)
axes[0].set_title("Escore do Isolation Forest — AK-47 Asiimov")
axes[0].set_xlabel("Data")
axes[0].set_ylabel("Escore de anomalia")
locator = mdates.AutoDateLocator(minticks=4, maxticks=7)
axes[0].xaxis.set_major_locator(locator)
axes[0].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

long_overlap = if_summary.melt(id_vars="short_name", value_vars=["eventos_if", "eventos_robustos", "sobreposicao"], var_name="tipo", value_name="eventos")
sns.barplot(data=long_overlap, y="short_name", x="eventos", hue="tipo", ax=axes[1])
axes[1].set_title("Comparação entre detectores")
axes[1].set_xlabel("Quantidade de eventos")
axes[1].set_ylabel("")
plt.tight_layout()
plt.savefig(FIG_DIR / "16_isolation_forest.png", dpi=160, bbox_inches="tight")
plt.show()
""")

md(r"""
**Como explicar:** Isolation Forest combina quatro features padronizadas por item: retorno, nível de volume, mudança de volume e volatilidade móvel. A amplitude diária foi excluída porque sua mediana e MAD são zero nesta amostra, tornando-a inadequada nesta forma. `contamination=0.02` força aproximadamente 2% das observações a serem marcadas, então a quantidade não mede prevalência real. A sobreposição mostra concordância parcial com o baseline, não acurácia. O método não conhece a ordem temporal diretamente.
""")

md("## 19. Ranking integrado de eventos para inspeção")
code(r"""
merged = df.merge(model_data[["market_hash_name", "date", "if_score", "if_candidate"]], on=["market_hash_name", "date"], how="left")
merged["if_score_rank"] = merged["if_score"].rank(pct=True)
merged["robust_score_rank"] = merged["robust_score"].rank(pct=True)
# skipna=False: linhas sem escore do Isolation Forest não entram no ranking com nota de um só método.
merged["integrated_rank"] = merged[["if_score_rank", "robust_score_rank"]].mean(axis=1, skipna=False)
inspection = merged.nlargest(25, "integrated_rank")[[
    "date", "short_name", "price_median", "volume", "log_return", "robust_score", "if_score", "robust_candidate", "if_candidate", "integrated_rank"
]]
display(inspection)
inspection.to_csv(HERE / "output" / "eventos_para_inspecao_notebook.csv", index=False)
""")

md(r"""
**Como explicar:** o ranking combina posição relativa no baseline e no Isolation Forest. Ele serve para escolher quais datas investigar primeiro. Não é probabilidade, não é rótulo e não substitui contexto externo.
""")

md("## 20. Validação contextual: atualização do Trade Up Contract em 22/10/2025")
code(r"""
EVENT_DATE = pd.Timestamp("2025-10-22")
window = df[(df["date"] >= EVENT_DATE - pd.Timedelta(days=30)) & (df["date"] <= EVENT_DATE + pd.Timedelta(days=30))].copy()

event_rows = []
fig, ax = plt.subplots(figsize=(12, 6))
for name, g in window.groupby("short_name"):
    g = g.sort_values("date").copy()
    # Base em D-1: o preço diário de 22/10 já pode conter parte da reação à atualização.
    before = g[g["date"] < EVENT_DATE]
    after = g[g["date"] >= EVENT_DATE]
    if before.empty or after.empty:
        continue
    base_row = before.iloc[-1]
    base = base_row["price_median"]
    price_at = lambda days: g[g["date"] <= EVENT_DATE + pd.Timedelta(days=days)].iloc[-1]["price_median"]
    g["event_index"] = 100 * g["price_median"] / base
    ax.plot(g["date"], g["event_index"], linewidth=1.3, label=name)
    # Volume: compara mediana com mediana (e máximo com máximo), nunca mediana com máximo.
    event_rows.append({
        "item": name,
        "data_base": base_row["date"].date(),
        "preco_base": base,
        "retorno_d0_%": 100 * (price_at(0) / base - 1),
        "retorno_7d_%": 100 * (price_at(7) / base - 1),
        "retorno_30d_%": 100 * (price_at(30) / base - 1),
        "volume_mediano_antes": before["volume"].median(),
        "volume_mediano_depois": after["volume"].median(),
        "razao_volume_mediano": after["volume"].median() / before["volume"].median(),
        "volume_maximo_antes": before["volume"].max(),
        "volume_maximo_depois": after["volume"].max(),
    })
ax.axvline(EVENT_DATE, color="#C00000", linestyle="--", linewidth=1.5, label="Atualização oficial — 22/10/2025")
ax.axhline(100, color="black", linestyle=":", linewidth=0.8)
ax.set_title("Janela de evento: 30 dias antes e depois da atualização do Trade Up Contract")
ax.set_xlabel("Data")
ax.set_ylabel("Índice de preço — base 100 em 21/10/2025 (D−1)")
ax.legend(fontsize=8)
locator = mdates.AutoDateLocator(minticks=5, maxticks=9)
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
plt.tight_layout()
plt.savefig(FIG_DIR / "17_evento_trade_up_2025-10-22.png", dpi=160, bbox_inches="tight")
plt.show()

event_summary = pd.DataFrame(event_rows)
display(event_summary)
event_summary.to_csv(HERE / "output" / "resumo_evento_trade_up_2025-10-22.csv", index=False)
""")

md(r"""
**Contexto oficial:** em 22/10/2025, uma atualização do Counter-Strike 2 ampliou o Trade Up Contract para permitir trocar cinco itens Covert por faca ou luvas. Fonte oficial: [Counter-Strike 2 Update — 22/10/2025](https://steamstore-a.akamaihd.net/news/externalpost/steam_community_announcements/1814309641424632).

**Como explicar:** o preço-base é o de 21/10/2025 (D−1), porque o preço diário de 22/10 pode já conter parte da reação; retornos de 7 e 30 dias são medidos a partir dessa base. O volume compara mediana antes com mediana depois (e máximo com máximo), para não inflar o efeito. Vários eventos de maior escore aparecem logo após a atualização e em diferentes skins. Isso favorece a hipótese de choque comum do ecossistema, não de manipulação isolada. A janela é descritiva e não separa todos os efeitos causais da atualização.
""")

md("## 21. Conclusões da etapa exploratória")
code(r"""
final_summary = {
    "linhas_dataset_completo": audit["rows"],
    "itens_historicos": audit["distinct_items"],
    "linhas_cesta": len(df),
    "periodo_cesta": [str(df["date"].min().date()), str(df["date"].max().date())],
    "eventos_robustos_limiar_4": int(df["robust_candidate"].sum()),
    "eventos_isolation_forest": int(model_data["if_candidate"].sum()),
    "pontos_pelt_penalizacao_10": len(pelt_breaks[10]),
    "aviso": "Nenhum resultado confirma manipulação.",
}
print(json.dumps(final_summary, ensure_ascii=False, indent=2))
(HERE / "output" / "resumo_notebook.json").write_text(json.dumps(final_summary, ensure_ascii=False, indent=2), encoding="utf-8")
""")

md(r"""
## Síntese para explicar ao professor

1. O dataset é grande e estruturalmente consistente, mas ainda é uma fonte de terceiro.
2. A cesta piloto possui cobertura quase diária e itens com perfis diferentes.
3. Preço, volume, retorno e volatilidade mostram comportamentos temporais distintos.
4. Padrões de calendário e correlações são descritivos, não causais.
5. O baseline robusto localiza extremos com regra transparente.
6. PELT localiza mudanças de regime e depende da penalização.
7. Isolation Forest combina atributos, mas a taxa marcada depende da contaminação escolhida.
8. Concordância entre métodos aumenta prioridade de inspeção, não confirma fraude.
9. A próxima etapa é validar dados, eventos externos e estabilidade dos resultados.

## O que ainda precisa ser decidido

- critérios da cesta final;
- validação externa do dataset;
- período principal;
- features finais;
- limiares e penalizações;
- protocolo de eventos legítimos;
- método de avaliação sem rótulos.
""")

nb["cells"] = cells
nbf.write(nb, OUT)
print(f"Notebook criado: {OUT} | células: {len(cells)}")
