#!/usr/bin/env python3
"""Regenera as tabelas do notebook a partir da cesta salva, sem o Parquet de 283 MB.

Entrada: output/cesta_2024_2026.csv (4.484 linhas: 5 skins Field-Tested, 2024-01-01 a 2026-06-15).
Usa apenas market_hash_name, date, price_median e volume; todas as variáveis derivadas
são recalculadas exatamente como em build_analysis_notebook.py (seções 5, 9, 12, 14-20),
já com as correções:
  - log_volume_change recebe a mesma regra de lacuna (> 3 dias) que log_return;
  - ranking integrado com mean(skipna=False);
  - evento de 22/10/2025 com base em D-1 e comparação de volume mediana × mediana.

Uso: .venv/bin/python gerar_tabelas_cesta.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruptures as rpt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.ensemble import IsolationForest

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({"figure.figsize": (11, 5.5), "axes.titlesize": 13, "axes.labelsize": 11})

HERE = Path(__file__).parent
OUT = HERE / "output"
FIG_DIR = HERE / "notebook_figures"
FIG_DIR.mkdir(exist_ok=True)
SELECTED_ITEMS = [
    "AK-47 | Asiimov (Field-Tested)",
    "AWP | Hyper Beast (Field-Tested)",
    "M4A1-S | Printstream (Field-Tested)",
    "Desert Eagle | Printstream (Field-Tested)",
    "Glock-18 | Water Elemental (Field-Tested)",
]
THRESHOLD = 4.0
EVENT_DATE = pd.Timestamp("2025-10-22")


def robust_z(series):
    median = series.median(skipna=True)
    mad = (series - median).abs().median(skipna=True)
    if not np.isfinite(mad) or mad == 0:
        return pd.Series(np.nan, index=series.index)
    return (series - median) / (1.4826 * mad)


# --- Seção 5: preparação (recalculada do zero a partir das colunas brutas) ---
df = pd.read_csv(OUT / "cesta_2024_2026.csv", usecols=["market_hash_name", "date", "price_median", "volume"])
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
df["rolling_vol_30"] = df.groupby("market_hash_name")["log_return"].transform(lambda s: s.rolling(30, min_periods=15).std())

# --- Seção 9: estatísticas dos retornos ---
return_stats = df.groupby("short_name")["log_return"].agg(["count", "mean", "std", "skew", "min", "max"])
return_stats[["mean", "std", "min", "max"]] *= 100
return_stats = return_stats.rename(columns={"mean": "média_%", "std": "desvio_%", "min": "mínimo_%", "max": "máximo_%"})
return_stats.to_csv(OUT / "estatisticas_retornos.csv")

# --- Seção 12: Spearman retorno × mudança de volume ---
correlations = []
for name, g in df.groupby("short_name"):
    valid = g[["log_return", "log_volume_change"]].dropna()
    rho, pvalue = spearmanr(valid["log_return"], valid["log_volume_change"])
    correlations.append({"item": name, "rho_spearman": rho, "p_value": pvalue, "n": len(valid)})
corr_df = pd.DataFrame(correlations).sort_values("rho_spearman")
corr_df.to_csv(OUT / "correlacao_retorno_volume.csv", index=False)

# --- Seção 14: correlação entre retornos das skins ---
return_corr = df.pivot(index="date", columns="short_name", values="log_return").corr(method="spearman")
return_corr.to_csv(OUT / "correlacao_retornos_skins.csv")

# --- Seção 15: baseline robusto (MAD) ---
for column, output in [
    ("log_return", "z_return"),
    ("log_volume", "z_volume"),
    ("log_volume_change", "z_volume_change"),
    ("rolling_vol_30", "z_rolling_vol"),
]:
    df[output] = df.groupby("market_hash_name")[column].transform(robust_z)
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
candidate_counts.to_csv(OUT / "contagem_eventos_mad.csv")

# --- Seção 16: sensibilidade do limiar ---
base_score = df[["z_return", "z_volume"]].abs().max(axis=1)
sensitivity_df = pd.DataFrame([
    {"limiar": t, "eventos": int((base_score >= t).sum()), "percentual": 100 * (base_score >= t).mean()}
    for t in [3.0, 3.5, 4.0, 4.5, 5.0, 6.0]
])
sensitivity_df.to_csv(OUT / "sensibilidade_limiar_mad.csv", index=False)

# --- Seção 17: PELT (AK-47) e sensibilidade à penalização ---
f = df[df["market_hash_name"] == SELECTED_ITEMS[0]].copy().reset_index(drop=True)
signal = np.column_stack([
    (f["log_price"] - f["log_price"].mean()) / f["log_price"].std(),
    (f["log_volume"] - f["log_volume"].mean()) / f["log_volume"].std(),
])
pelt_rows, pelt_breaks = [], {}
for penalty in [5, 8, 10, 12, 15, 20]:
    breaks = rpt.Pelt(model="rbf", min_size=14, jump=1).fit(signal).predict(pen=penalty)
    breaks = [b for b in breaks if b < len(f)]
    pelt_breaks[penalty] = breaks
    pelt_rows.append({"penalizacao": penalty, "pontos_mudanca": len(breaks)})
pelt_sensitivity = pd.DataFrame(pelt_rows)
pelt_sensitivity.to_csv(OUT / "sensibilidade_pelt.csv", index=False)
pelt_points = pd.DataFrame([
    {"penalizacao": p, "ordem": i + 1, "indice": b, "data": f.iloc[b]["date"].date()}
    for p, bs in pelt_breaks.items() for i, b in enumerate(bs)
])
pelt_points.to_csv(OUT / "pontos_mudanca_pelt_notebook.csv", index=False)

# --- Seção 18: Isolation Forest ---
FEATURES = ["z_return", "z_volume", "z_volume_change", "z_rolling_vol"]
model_data = df.dropna(subset=FEATURES).copy()
iso = IsolationForest(n_estimators=300, contamination=0.02, random_state=42, n_jobs=-1)
model_data["if_label"] = iso.fit_predict(model_data[FEATURES])
model_data["if_score"] = -iso.decision_function(model_data[FEATURES])
model_data["if_candidate"] = model_data["if_label"] == -1
model_data["both"] = model_data["if_candidate"] & model_data["robust_candidate"]
if_summary = model_data.groupby("short_name").agg(
    observacoes=("date", "size"),
    eventos_if=("if_candidate", "sum"),
    eventos_robustos=("robust_candidate", "sum"),
    sobreposicao=("both", "sum"),
).reset_index()
if_summary.to_csv(OUT / "resumo_isolation_forest.csv", index=False)

# --- Seção 19: ranking integrado ---
merged = df.merge(model_data[["market_hash_name", "date", "if_score", "if_candidate"]], on=["market_hash_name", "date"], how="left")
merged["if_score_rank"] = merged["if_score"].rank(pct=True)
merged["robust_score_rank"] = merged["robust_score"].rank(pct=True)
merged["integrated_rank"] = merged[["if_score_rank", "robust_score_rank"]].mean(axis=1, skipna=False)
inspection = merged.nlargest(25, "integrated_rank")[[
    "date", "short_name", "price_median", "volume", "log_return", "robust_score", "if_score", "robust_candidate", "if_candidate", "integrated_rank"
]]
inspection.to_csv(OUT / "eventos_para_inspecao_notebook.csv", index=False)

# --- Seção 20: janela do evento Trade Up (base D-1) + figura 17 ---
window = df[(df["date"] >= EVENT_DATE - pd.Timedelta(days=30)) & (df["date"] <= EVENT_DATE + pd.Timedelta(days=30))]
event_rows = []
fig, ax = plt.subplots(figsize=(12, 6))
for name, g in window.groupby("short_name"):
    g = g.sort_values("date").copy()
    before = g[g["date"] < EVENT_DATE]
    after = g[g["date"] >= EVENT_DATE]
    if before.empty or after.empty:
        continue
    base_row = before.iloc[-1]
    base = base_row["price_median"]
    price_at = lambda days: g[g["date"] <= EVENT_DATE + pd.Timedelta(days=days)].iloc[-1]["price_median"]
    g["event_index"] = 100 * g["price_median"] / base
    ax.plot(g["date"], g["event_index"], linewidth=1.3, label=name)
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
plt.close(fig)
event_summary = pd.DataFrame(event_rows)
event_summary.to_csv(OUT / "resumo_evento_trade_up_2025-10-22.csv", index=False)

# --- Resumo no terminal ---
print(f"Linhas: {len(df)} | período: {df['date'].min().date()} a {df['date'].max().date()} | maior lacuna: {df['gap_days'].max():.0f} dias")
print(f"Linhas com lacuna > 3 dias: {int((df['gap_days'] > 3).sum())}")
print(f"Linhas no Isolation Forest: {len(model_data)} de {len(df)} | eventos IF: {int(model_data['if_candidate'].sum())}")
print(f"Linhas sem if_score (fora do ranking): {int(merged['if_score'].isna().sum())}")
print("Arquivos gravados em", OUT)
