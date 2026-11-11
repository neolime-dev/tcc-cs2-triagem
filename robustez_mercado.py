#!/usr/bin/env python3
"""Índice de mercado e baseline ajustado ao mercado (item 4 da arguição simulada).

Entrada: daily_prices.parquet (Kaggle, versão 4) — baixar antes com download_data.py.
Saída:   output/robustez/mercado_*.csv e output/robustez/resumo_mercado.json.
Não altera nenhum arquivo existente.

Ideia: manipulação de um item é movimento PRÓPRIO do item. Por isso, o movimento comum do
mercado (mediana diária entre centenas de skins) é subtraído antes de aplicar o mesmo escore
robusto do TCC. O que sobra é o desvio do item em relação ao mercado.

Uso: .venv/bin/python robustez_mercado.py
"""
import datetime as dt
import json
from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

HERE = Path(__file__).parent
OUT = HERE / "output" / "robustez"
OUT.mkdir(parents=True, exist_ok=True)
START, END = dt.date(2024, 1, 1), dt.date(2026, 6, 15)
THRESHOLD = 4.0
SHOCK_START = pd.Timestamp("2025-10-23")
BASKET = [
    "AK-47 | Asiimov (Field-Tested)",
    "AWP | Hyper Beast (Field-Tested)",
    "M4A1-S | Printstream (Field-Tested)",
    "Desert Eagle | Printstream (Field-Tested)",
    "Glock-18 | Water Elemental (Field-Tested)",
]
MIN_DAYS, MIN_MEDIAN_VOLUME = 880, 30  # universo do índice: cobertura quase diária e liquidez mínima


def robust_z(frame):
    """Escore robusto por coluna (mediana e MAD da janela completa), como no TCC."""
    med = frame.median()
    mad = (frame - med).abs().median()
    return (frame - med) / (1.4826 * mad.where(mad > 0))


def n_episodes(flags, max_gap=2):
    total = 0
    for col in flags.columns:
        d = flags.index[flags[col].to_numpy()]
        if len(d):
            total += int((pd.Series(d).diff().dt.days > max_gap).sum() + 1)
    return total


path = Path(kagglehub.dataset_download("devynpruden/cs2-skin-price-history-2013-2026/versions/4"))
table = ds.dataset(path / "daily_prices.parquet").to_table(
    filter=(ds.field("date") >= START) & (ds.field("date") <= END),
    columns=["market_hash_name", "date", "price_median", "volume", "source_point_count"],
).to_pandas()
table["date"] = pd.to_datetime(table["date"])
resumo = {}

# --- Checagem de dado: quantos pontos da fonte formam cada preço diário da cesta ---
b = table[table["market_hash_name"].isin(BASKET)]
resumo["pontos_da_fonte_cesta"] = {
    "pct_dias_com_ponto_unico": round(100 * (b["source_point_count"] == 1).mean(), 1),
    "primeiro_dia_com_mais_de_um_ponto": str(b.loc[b["source_point_count"] > 1, "date"].min().date()),
    "dias_com_mais_de_um_ponto": int((b["source_point_count"] > 1).sum()),
}

# --- Universo e matrizes data × item ---
stats = table.groupby("market_hash_name").agg(n=("date", "size"), vol=("volume", "median"))
universe = stats[(stats["n"] >= MIN_DAYS) & (stats["vol"] >= MIN_MEDIAN_VOLUME)].index.difference(BASKET)
wide = table[table["market_hash_name"].isin(universe.union(BASKET))]
price = wide.pivot(index="date", columns="market_hash_name", values="price_median")
volume = wide.pivot(index="date", columns="market_hash_name", values="volume")
days = pd.Series(price.index, index=price.index)


def log_returns(p):
    out = {}
    for col in p.columns:
        s = np.log(p[col].dropna())
        r = s.diff()
        r[days.loc[s.index].diff().dt.days > 3] = np.nan
        out[col] = r
    return pd.DataFrame(out).reindex(p.index)


ret = log_returns(price)
logv = np.log1p(volume)
market_ret = ret[universe].median(axis=1)                         # retorno comum do dia
market_vol = (logv[universe] - logv[universe].median()).median(axis=1)  # desvio comum do log-volume
index = pd.DataFrame({"retorno_mercado": market_ret, "volume_mercado": market_vol,
                      "itens_com_retorno": ret[universe].notna().sum(axis=1)})
index["z_retorno_mercado"] = robust_z(index[["retorno_mercado"]])["retorno_mercado"]
index["z_volume_mercado"] = robust_z(index[["volume_mercado"]])["volume_mercado"]
resumo["universo"] = {"itens_no_indice": len(universe), "criterio": f">= {MIN_DAYS} dias e volume mediano >= {MIN_MEDIAN_VOLUME}"}

# --- Amplitude: fração do universo marcada pelo MESMO baseline do TCC em cada dia ---
zr_u, zv_u = robust_z(ret[universe]), robust_z(logv[universe])
cand_u = (zr_u.abs() >= THRESHOLD) | (zv_u.abs() >= THRESHOLD)
index["pct_universo_marcado"] = 100 * cand_u.mean(axis=1)
index.round(5).to_csv(OUT / "mercado_indice_diario.csv")
top_breadth = index["pct_universo_marcado"].nlargest(15).round(1)
top_breadth.rename_axis("data").to_csv(OUT / "mercado_dias_de_maior_amplitude.csv")
rate_item = cand_u.mean() * 100
resumo["amplitude"] = {
    "taxa_candidatos_universo_%": round(100 * cand_u.to_numpy().mean(), 2),
    "taxa_por_item_p10_mediana_p90_%": [round(float(rate_item.quantile(q)), 2) for q in (0.1, 0.5, 0.9)],
    "pct_universo_marcado_mediana_diaria": round(float(index["pct_universo_marcado"].median()), 2),
    "dias_com_mais_de_10pct_do_universo_marcado": int((index["pct_universo_marcado"] > 10).sum()),
    "pct_marcado_em_2024-10-03": round(float(index.loc["2024-10-03", "pct_universo_marcado"]), 1),
    "pct_marcado_em_2025-10-23": round(float(index.loc["2025-10-23", "pct_universo_marcado"]), 1),
    "pct_marcado_em_2025-02-01": round(float(index.loc["2025-02-01", "pct_universo_marcado"]), 1),
    "pct_marcado_em_2025-06-24": round(float(index.loc["2025-06-24", "pct_universo_marcado"]), 1),
    "pct_dos_candidatos_do_universo_apos_23out2025": round(100 * cand_u[cand_u.index >= SHOCK_START].to_numpy().sum() / cand_u.to_numpy().sum(), 1),
}

# --- Baseline da cesta: original × ajustado ao mercado ---
zr, zv = robust_z(ret[BASKET]), robust_z(logv[BASKET])
score = pd.concat([zr.abs(), zv.abs()]).groupby(level=0).max()
cand = score >= THRESHOLD
assert int(cand.to_numpy().sum()) == 111, "baseline diverge do TCC"
zr_a = robust_z(ret[BASKET].sub(market_ret, axis=0))
zv_a = robust_z(logv[BASKET].sub(market_vol, axis=0))
score_a = pd.concat([zr_a.abs(), zv_a.abs()]).groupby(level=0).max()
cand_a = score_a >= THRESHOLD
after = cand.index >= SHOCK_START
rows = []
for name in BASKET + ["TOTAL"]:
    sel = BASKET if name == "TOTAL" else [name]
    rows.append({
        "item": name.replace(" (Field-Tested)", ""),
        "original": int(cand[sel].to_numpy().sum()), "ajustado": int(cand_a[sel].to_numpy().sum()),
        "em_comum": int((cand[sel] & cand_a[sel]).to_numpy().sum()),
        "original_apos_23out2025": int(cand[sel][after].to_numpy().sum()),
        "ajustado_apos_23out2025": int(cand_a[sel][after].to_numpy().sum()),
        "ajustado_retorno": int((zr_a[sel].abs() >= THRESHOLD).to_numpy().sum()),
        "ajustado_volume": int((zv_a[sel].abs() >= THRESHOLD).to_numpy().sum()),
        "episodios_original": n_episodes(cand[sel]), "episodios_ajustado": n_episodes(cand_a[sel]),
    })
pd.DataFrame(rows).to_csv(OUT / "mercado_baseline_original_vs_ajustado.csv", index=False)
resumo["baseline_ajustado"] = rows[-1]

def tall(frame, name):
    return frame[BASKET].rename_axis(index="data", columns="item").stack(future_stack=True).rename(name)


long = pd.concat([
    tall(score, "escore_original"), tall(score_a, "escore_ajustado"), tall(ret, "retorno"),
    tall(volume, "volume"), tall(zr_a, "z_retorno_ajustado"), tall(zv_a, "z_volume_ajustado"),
], axis=1).reset_index()
long["retorno_mercado"] = long["data"].map(market_ret)
long = long.dropna(subset=["volume"])
long["item"] = long["item"].str.replace(" (Field-Tested)", "", regex=False)
top = long.nlargest(25, "escore_ajustado").sort_values(["data", "item"])
top.round(3).to_csv(OUT / "mercado_top25_ajustado.csv", index=False)
resumo["top25_ajustado"] = {
    "apos_23out2025": int((top["data"] >= SHOCK_START).sum()),
    "datas_distintas": int(top["data"].nunique()),
    "por_item": top.groupby("item").size().to_dict(),
}
resumo["choques_do_proprio_indice"] = {
    "dias_com_z_retorno_mercado>=4": [str(d.date()) for d in index.index[index["z_retorno_mercado"].abs() >= THRESHOLD]],
    "dias_com_z_volume_mercado>=4": int((index["z_volume_mercado"].abs() >= THRESHOLD).sum()),
}
resumo["eventos_citados_no_tcc"] = {
    d: {"retorno_mercado_%": round(100 * float(market_ret.loc[d]), 2),
        "z_retorno_mercado": round(float(index.loc[d, "z_retorno_mercado"]), 1),
        "z_volume_mercado": round(float(index.loc[d, "z_volume_mercado"]), 1)}
    for d in ["2024-10-03", "2025-02-01", "2025-06-24", "2025-10-22", "2025-10-23"]
}
(OUT / "resumo_mercado.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
print(json.dumps(resumo, ensure_ascii=False, indent=2, default=str))
print(top_breadth.to_string())
print(top[["data", "item", "retorno", "retorno_mercado", "volume", "escore_original", "escore_ajustado"]].round(3).to_string(index=False))
