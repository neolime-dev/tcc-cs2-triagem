#!/usr/bin/env python3
"""Testes de robustez pedidos na arguição simulada (itens 1, 2, 3, 5, 6 e 7).

Entrada: output/cesta_2024_2026.csv (mesma cesta do TCC).
Saída:   output/robustez/*.csv e output/robustez/resumo_robustez.json.
Não altera nenhum arquivo existente. As definições (retorno, escore robusto, PELT,
Isolation Forest, ranking integrado) são as mesmas de gerar_tabelas_cesta.py.

Uso: .venv/bin/python robustez_banca.py [--sem-sinteticas]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import ruptures as rpt
from sklearn.ensemble import IsolationForest

HERE = Path(__file__).parent
OUT = HERE / "output" / "robustez"
OUT.mkdir(parents=True, exist_ok=True)
THRESHOLD = 4.0
PENALTIES = [5, 8, 10, 12, 15, 20]
FEATURES = ["z_return", "z_volume", "z_volume_change", "z_rolling_vol"]
SHOCK_START = pd.Timestamp("2025-10-23")
resumo = {}


def robust_z(series):
    median = series.median(skipna=True)
    mad = (series - median).abs().median(skipna=True)
    if not np.isfinite(mad) or mad == 0:
        return pd.Series(np.nan, index=series.index)
    return (series - median) / (1.4826 * mad)


def rolling_z(series, window, min_periods):
    """Escore robusto com mediana e MAD apenas dos `window` dias ANTERIORES (sem dados futuros)."""
    assert min_periods == window  # só janelas completas; uma ausência na janela deixa o escore indefinido
    x = series.to_numpy(float)
    out = np.full(len(x), np.nan)
    if len(x) > window:
        w = np.lib.stride_tricks.sliding_window_view(x, window)[:-1]  # w[i] = x[i:i+window], referência de x[i+window]
        med = np.median(w, axis=1)
        mad = np.median(np.abs(w - med[:, None]), axis=1)
        mad[~(mad > 0)] = np.nan
        out[window:] = (x[window:] - med) / (1.4826 * mad)
    return pd.Series(out, index=series.index)


def prepare(raw):
    df = raw.sort_values(["market_hash_name", "date"]).reset_index(drop=True)
    g = df.groupby("market_hash_name")
    df["gap_days"] = g["date"].diff().dt.days
    df["log_price"] = np.log(df["price_median"])
    df["log_return"] = g["log_price"].diff()
    df.loc[df["gap_days"] > 3, "log_return"] = np.nan
    df["log_volume"] = np.log1p(df["volume"])
    df["log_volume_change"] = g["log_volume"].diff()
    df.loc[df["gap_days"] > 3, "log_volume_change"] = np.nan
    df["rolling_vol_30"] = df.groupby("market_hash_name")["log_return"].transform(
        lambda s: s.rolling(30, min_periods=15).std()
    )
    for col, out in [("log_return", "z_return"), ("log_volume", "z_volume"),
                     ("log_volume_change", "z_volume_change"), ("rolling_vol_30", "z_rolling_vol")]:
        df[out] = df.groupby("market_hash_name")[col].transform(robust_z)
    df["robust_score"] = df[["z_return", "z_volume"]].abs().max(axis=1)
    df["robust_candidate"] = df["robust_score"] >= THRESHOLD
    return df


def fit_if(df, seed=42, contamination=0.02):
    data = df.dropna(subset=FEATURES)
    iso = IsolationForest(n_estimators=300, contamination=contamination, random_state=seed, n_jobs=-1)
    label = iso.fit_predict(data[FEATURES])
    score = pd.Series(-iso.decision_function(data[FEATURES]), index=data.index)
    return score.reindex(df.index), pd.Series(label == -1, index=data.index).reindex(df.index, fill_value=False)


def integrated_top(df, if_score, k=25):
    rank = pd.concat([if_score.rank(pct=True), df["robust_score"].rank(pct=True)], axis=1).mean(axis=1, skipna=False)
    return set(rank.nlargest(k).index)


def episodes(df, flag_col, score_col, max_gap=2):
    """Agrupa item-dias marcados em episódios: dias do mesmo item separados por até `max_gap` dias."""
    rows = []
    for name, g in df[df[flag_col]].groupby("short_name"):
        block = (g["date"].diff().dt.days > max_gap).cumsum()
        for _, e in g.groupby(block):
            rows.append({"item": name, "inicio": e["date"].min().date(), "fim": e["date"].max().date(),
                         "dias_marcados": len(e), "escore_max": round(e[score_col].abs().max(), 2)})
    return pd.DataFrame(rows).sort_values(["inicio", "item"]).reset_index(drop=True)


raw = pd.read_csv(HERE / "output" / "cesta_2024_2026.csv",
                  usecols=["market_hash_name", "date", "price_median", "volume"], parse_dates=["date"])
raw["short_name"] = raw["market_hash_name"].str.replace(" (Field-Tested)", "", regex=False)
df = prepare(raw)
items = list(df["short_name"].unique())
assert int(df["robust_candidate"].sum()) == 111, "baseline diverge do TCC"

# ---------------- Item 1: PELT nos cinco itens ----------------
pelt_rows = []
for name, f in df.groupby("short_name"):
    f = f.reset_index(drop=True)
    signal = np.column_stack([(f[c] - f[c].mean()) / f[c].std() for c in ["log_price", "log_volume"]])
    algo = rpt.Pelt(model="rbf", min_size=14, jump=1).fit(signal)
    for pen in PENALTIES:
        for b in [b for b in algo.predict(pen=pen) if b < len(f)]:
            pelt_rows.append({"item": name, "penalizacao": pen, "data": f.loc[b, "date"]})
pelt = pd.DataFrame(pelt_rows)
pelt.assign(data=pelt["data"].dt.date).to_csv(OUT / "pelt_cinco_itens_pontos.csv", index=False)
pelt.groupby(["item", "penalizacao"]).size().unstack().to_csv(OUT / "pelt_cinco_itens_contagem.csv")
persist = []
for name, g in pelt.groupby("item"):
    for d in sorted(g[g["penalizacao"] == PENALTIES[-1]]["data"]):  # pontos da penalização mais alta
        exact = all((g[g["penalizacao"] == p]["data"] == d).any() for p in PENALTIES)
        near = all(((g[g["penalizacao"] == p]["data"] - d).abs().dt.days <= 7).any() for p in PENALTIES)
        persist.append({"item": name, "data": d.date(), "em_todas_exato": exact, "em_todas_ate_7_dias": near})
persist = pd.DataFrame(persist)
persist.to_csv(OUT / "pelt_cinco_itens_persistentes.csv", index=False)
resumo["pelt"] = {
    "pontos_por_item_pen10": pelt[pelt["penalizacao"] == 10].groupby("item").size().to_dict(),
    "persistentes_exatos": persist[persist["em_todas_exato"]].groupby("item")["data"].apply(lambda s: [str(x) for x in s]).to_dict(),
    "persistentes_ate_7_dias": persist[persist["em_todas_ate_7_dias"]].groupby("item")["data"].apply(lambda s: [str(x) for x in s]).to_dict(),
}

# ---------------- Item 2: Isolation Forest — sementes e contaminação ----------------
ref_score, ref_flag = fit_if(df, seed=42, contamination=0.02)
ref_top = integrated_top(df, ref_score)
assert int(ref_flag.sum()) == 89, "Isolation Forest diverge do TCC"
seed_rows, flags_by = [], {}
for cont in [0.01, 0.02, 0.05]:
    for seed in range(10):
        score, flag = fit_if(df, seed=seed, contamination=cont)
        flags_by[(cont, seed)] = set(flag[flag].index)
        seed_rows.append({
            "contaminacao": cont, "semente": seed, "marcados": int(flag.sum()),
            "tambem_baseline": int((flag & df["robust_candidate"]).sum()),
            "top25_em_comum_com_referencia": len(integrated_top(df, score) & ref_top),
            "marcados_em_comum_com_referencia_c002_s42": len(flags_by[(cont, seed)] & set(ref_flag[ref_flag].index)),
        })
seeds = pd.DataFrame(seed_rows)
seeds.to_csv(OUT / "if_sementes_contaminacao.csv", index=False)
stable = {}
for cont in [0.01, 0.02, 0.05]:
    sets = [flags_by[(cont, s)] for s in range(10)]
    stable[str(cont)] = {
        "marcados_min_max": [min(map(len, sets)), max(map(len, sets))],
        "marcados_em_todas_as_10_sementes": len(set.intersection(*sets)),
        "marcados_em_alguma_semente": len(set.union(*sets)),
        "tambem_baseline_min_max": [int(seeds[seeds.contaminacao == cont]["tambem_baseline"].min()),
                                    int(seeds[seeds.contaminacao == cont]["tambem_baseline"].max())],
    }
c02 = seeds[seeds.contaminacao == 0.02]["top25_em_comum_com_referencia"]
resumo["isolation_forest"] = {"por_contaminacao": stable,
                              "top25_em_comum_com_referencia_min_max_10_sementes": [int(c02.min()), int(c02.max())]}

# ---------------- Item 3: referência em janela móvel (sem dados futuros) ----------------
roll_rows = []
for window in [60, 90, 180]:
    zr = df.groupby("market_hash_name")["log_return"].transform(lambda s: rolling_z(s, window, window))
    zv = df.groupby("market_hash_name")["log_volume"].transform(lambda s: rolling_z(s, window, window))
    score = pd.concat([zr, zv], axis=1).abs().max(axis=1)
    elig = zr.notna() & zv.notna()
    cand = (score >= THRESHOLD) & elig
    df[f"roll{window}_score"], df[f"roll{window}_cand"] = score, cand
    glob = df["robust_candidate"] & elig
    roll_rows.append({
        "janela_dias": window, "obs_avaliadas": int(elig.sum()),
        "candidatos_movel": int(cand.sum()), "pct_movel": round(100 * cand.sum() / elig.sum(), 2),
        "candidatos_global_mesmas_obs": int(glob.sum()), "em_comum": int((cand & glob).sum()),
        "movel_apos_23out2025": int((cand & (df["date"] >= SHOCK_START)).sum()),
        "global_apos_23out2025": int((glob & (df["date"] >= SHOCK_START)).sum()),
        "movel_retorno": int(((zr.abs() >= THRESHOLD) & elig).sum()),
        "movel_volume": int(((zv.abs() >= THRESHOLD) & elig).sum()),
        "episodios_movel": len(episodes(df, f"roll{window}_cand", f"roll{window}_score")),
    })
rolling = pd.DataFrame(roll_rows)
rolling.to_csv(OUT / "baseline_janela_movel.csv", index=False)
episodes(df, "roll90_cand", "roll90_score").to_csv(OUT / "episodios_janela_movel_90.csv", index=False)
resumo["janela_movel"] = roll_rows

# ---------------- Item 6: episódios do baseline global ----------------
ep = episodes(df, "robust_candidate", "robust_score")
ep.to_csv(OUT / "episodios_baseline_global.csv", index=False)
cand_dates = pd.Series(sorted(df.loc[df["robust_candidate"], "date"].unique()))
resumo["episodios"] = {
    "item_dias": 111, "episodios_por_item": len(ep), "datas_distintas": len(cand_dates),
    "blocos_de_datas_no_mercado": int((cand_dates.diff().dt.days > 2).sum() + 1),
    "episodios_por_item_detalhe": ep.groupby("item").size().to_dict(),
    "maior_episodio": ep.sort_values("dias_marcados").iloc[-1].astype(str).to_dict(),
}

# ---------------- Item 7: top 25 só do baseline × ranking integrado ----------------
has_if = ref_score.notna()
base_top = set(df[has_if]["robust_score"].nlargest(25).index)
if_top = set(ref_score.nlargest(25).index)
cols = ["date", "short_name", "log_return", "volume", "robust_score"]
cmp = df.loc[sorted(base_top | ref_top), cols].assign(if_score=ref_score)
cmp["no_top25_baseline"] = cmp.index.isin(base_top)
cmp["no_top25_integrado"] = cmp.index.isin(ref_top)
cmp.sort_values(["date", "short_name"]).to_csv(OUT / "top25_baseline_vs_integrado.csv", index=False)


def n_ep(idx):
    t = df.loc[sorted(idx)].assign(flag=True)
    return len(episodes(t, "flag", "robust_score"))


resumo["top25"] = {
    "baseline_e_integrado_em_comum": len(base_top & ref_top),
    "if_sozinho_e_integrado_em_comum": len(if_top & ref_top),
    "episodios_no_top25_baseline": n_ep(base_top), "episodios_no_top25_integrado": n_ep(ref_top),
    "top25_baseline_apos_23out2025": int((df.loc[sorted(base_top), "date"] >= SHOCK_START).sum()),
    "top25_integrado_apos_23out2025": int((df.loc[sorted(ref_top), "date"] >= SHOCK_START).sum()),
}

# ---------------- Item 5: anomalias sintéticas ----------------
if "--sem-sinteticas" not in sys.argv:
    rng = np.random.default_rng(2026)
    REPS, PER_ITEM = 20, 5
    MAGS = [(0.05, 2), (0.10, 3), (0.20, 5), (0.40, 10)]  # (alta de preço, multiplicador de volume)
    blocked = {}  # índices posicionais por item onde já existe candidato real (±3 dias)
    for name, g in df.groupby("short_name"):
        pos = np.flatnonzero(g["robust_candidate"].to_numpy() | g["roll90_cand"].to_numpy() | ref_flag.loc[g.index].to_numpy())
        blocked[name] = set(int(p + k) for p in pos for k in range(-4, 5))

    def sites(name, n, dur):
        chosen = []
        while len(chosen) < PER_ITEM:
            i = int(rng.integers(100, n - 10))
            window = set(range(i - 1, i + dur + 2))
            if window & blocked[name] or any(abs(i - c) < 40 for c in chosen):
                continue
            chosen.append(i)
        return chosen

    synth = []
    for kind in ["preco", "volume", "preco_e_volume"]:
        for dur in [1, 3]:
            for dp, mv in MAGS:
                hits = {m: {i: 0 for i in items} for m in ["global", "movel90", "if"]}
                for _ in range(REPS):
                    mod = raw.sort_values(["market_hash_name", "date"]).reset_index(drop=True).copy()
                    mod["price_median"] = mod["price_median"].astype(float)
                    mod["volume"] = mod["volume"].astype(float)
                    injected = []
                    for name, g in mod.groupby("short_name"):
                        for i in sites(name, len(g), dur):
                            idx = g.index[i:i + dur]
                            if kind != "volume":
                                mod.loc[idx, "price_median"] *= 1 + dp
                            if kind != "preco":
                                mod.loc[idx, "volume"] *= mv
                            injected.append((name, g.index[i], g.index[i + dur]))  # inclui o dia da reversão
                    m = prepare(mod)
                    zr = m.groupby("market_hash_name")["log_return"].transform(lambda s: rolling_z(s, 90, 90))
                    zv = m.groupby("market_hash_name")["log_volume"].transform(lambda s: rolling_z(s, 90, 90))
                    roll_c = pd.concat([zr, zv], axis=1).abs().max(axis=1) >= THRESHOLD
                    _, if_c = fit_if(m)
                    for name, a, b in injected:
                        hits["global"][name] += bool(m.loc[a:b, "robust_candidate"].any())
                        hits["movel90"][name] += bool(roll_c.loc[a:b].any())
                        hits["if"][name] += bool(if_c.loc[a:b].any())
                total = REPS * PER_ITEM
                for method, by_item in hits.items():
                    row = {"tipo": kind, "duracao_dias": dur, "alta_preco_%": 100 * dp if kind != "volume" else 0,
                           "mult_volume": mv if kind != "preco" else 1, "metodo": method,
                           "recuperacao_geral_%": round(100 * sum(by_item.values()) / (total * len(items)), 1)}
                    row.update({i: round(100 * by_item[i] / total, 0) for i in items})
                    synth.append(row)
                print("sintéticas:", kind, dur, dp, mv, flush=True)
    synth = pd.DataFrame(synth)
    synth.to_csv(OUT / "anomalias_sinteticas_recuperacao.csv", index=False)
    resumo["sinteticas"] = {"injecoes_por_configuracao": REPS * PER_ITEM * len(items),
                            "arquivo": "anomalias_sinteticas_recuperacao.csv"}

(OUT / "resumo_robustez.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
print(json.dumps(resumo, ensure_ascii=False, indent=2, default=str))
