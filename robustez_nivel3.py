#!/usr/bin/env python3
"""Regra testável para o nível 3 ("possível sinal de manipulação") e calendário de eventos públicos.

Entradas: output/cesta_2024_2026.csv e output/robustez/mercado_indice_diario.csv
          (gerado por robustez_mercado.py).
Saídas:   output/robustez/nivel3_*.csv, calendario_*.csv e resumo_nivel3.json.
Não altera nenhum arquivo existente.

Regra (todas as condições precisam valer para o episódio):
  C1 movimento próprio do item: nenhum dia do episódio com amplitude > 10% do universo;
  C2 alta conjunta: algum dia com escore de retorno >= +4 e algum dia com escore de volume >= +4;
  C3 reversão: em até 7 dias após o pico, o preço devolve ao menos metade da alta;
  C4 sem evento público documentado entre 7 dias antes do início e o fim do episódio.

Uso: .venv/bin/python robustez_nivel3.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
OUT = HERE / "output" / "robustez"
THRESHOLD, BREADTH_LIMIT, EVENT_WINDOW = 4.0, 10.0, 7

# Eventos públicos documentados. Atualizações: data UTC da publicação no canal oficial de notícias do jogo
# (todas entre 22h58 e 23h50 UTC). Promoções: data de início. Fontes no .bib do TCC.
CALENDAR = [
    ("2024-03-14", "promoção", "Steam Spring Sale 2024"),
    ("2024-06-27", "promoção", "Steam Summer Sale 2024"),
    ("2024-10-02", "atualização", "The Armory"),
    ("2024-11-27", "promoção", "Steam Autumn Sale 2024"),
    ("2024-12-19", "promoção", "Steam Winter Sale 2024"),
    ("2025-01-28", "atualização", "Premier Season Two"),
    ("2025-03-13", "promoção", "Steam Spring Sale 2025"),
    ("2025-03-31", "atualização", "Spring Forward (Fever Case)"),
    ("2025-06-26", "promoção", "Steam Summer Sale 2025"),
    ("2025-09-29", "promoção", "Steam Autumn Sale 2025"),
    ("2025-10-22", "atualização", "Trade Up Contract ampliado"),
    ("2025-11-12", "atualização", "Budapest Major Hub"),
    ("2025-12-18", "promoção", "Steam Winter Sale 2025"),
    ("2026-03-19", "promoção", "Steam Spring Sale 2026"),
]
cal = pd.DataFrame(CALENDAR, columns=["data", "tipo", "evento"])
cal["data"] = pd.to_datetime(cal["data"])


def robust_z(s):
    med = s.median()
    mad = (s - med).abs().median()
    return (s - med) / (1.4826 * mad) if mad > 0 else s * np.nan


df = pd.read_csv(HERE / "output" / "cesta_2024_2026.csv",
                 usecols=["market_hash_name", "date", "price_median", "volume"], parse_dates=["date"])
df = df.sort_values(["market_hash_name", "date"]).reset_index(drop=True)
df["item"] = df["market_hash_name"].str.replace(" (Field-Tested)", "", regex=False)
g = df.groupby("item")
df["gap"] = g["date"].diff().dt.days
df["ret"] = np.log(df["price_median"]).groupby(df["item"]).diff()
df.loc[df["gap"] > 3, "ret"] = np.nan
df["logv"] = np.log1p(df["volume"])
df["z_ret"] = df.groupby("item")["ret"].transform(robust_z)
df["z_vol"] = df.groupby("item")["logv"].transform(robust_z)
df["cand"] = df[["z_ret", "z_vol"]].abs().max(axis=1) >= THRESHOLD
assert int(df["cand"].sum()) == 111, "baseline diverge do TCC"

idx = pd.read_csv(OUT / "mercado_indice_diario.csv", parse_dates=["date"], index_col="date")
breadth = idx["pct_universo_marcado"]


def last_event(day, end=None):
    """Evento público entre EVENT_WINDOW dias antes de `day` e `end` (padrão: o próprio `day`)."""
    end = day if end is None else end
    hit = cal[(cal["data"] >= day - pd.Timedelta(days=EVENT_WINDOW)) & (cal["data"] <= end)]
    return (hit.iloc[-1]["evento"], int((day - hit.iloc[-1]["data"]).days)) if len(hit) else (None, None)


def evaluate(series, start, end, vol_thr=THRESHOLD, rev_days=7):
    ep = series[(series["date"] >= start) & (series["date"] <= end)]
    before = series[series["date"] < start]
    p_pre = before["price_median"].iloc[-1] if len(before) else np.nan
    peak = ep.loc[ep["price_median"].idxmax()]
    after = series[(series["date"] > peak["date"]) & (series["date"] <= peak["date"] + pd.Timedelta(days=rev_days))]
    rise = peak["price_median"] / p_pre - 1
    reverted = bool(rise > 0 and len(after) and after["price_median"].min() <= p_pre + 0.5 * (peak["price_median"] - p_pre))
    b = breadth.reindex(pd.date_range(start, end)).max()
    return {
        "amplitude_max_%": round(float(b), 1),
        "c1_proprio_do_item": bool(b <= BREADTH_LIMIT),
        "z_ret_max": round(float(ep["z_ret"].max()), 1), "z_ret_min": round(float(ep["z_ret"].min()), 1),
        "z_vol_max": round(float(ep["z_vol"].max()), 1),
        "c2_alta_conjunta": bool(ep["z_ret"].max() >= THRESHOLD and ep["z_vol"].max() >= vol_thr),
        "alta_ate_pico_%": round(100 * float(rise), 1),
        "c3_reversao": reverted,
    }


rows = []
for item, s in df.groupby("item"):
    c = s[s["cand"]]
    block = (c["date"].diff().dt.days > 2).cumsum()
    for _, e in c.groupby(block):
        start, end = e["date"].min(), e["date"].max()
        ev, lag = last_event(start, end)
        row = {"item": item, "inicio": start.date(), "fim": end.date(), "dias_marcados": len(e)}
        row.update(evaluate(s, start, end))
        row.update({"evento_publico": ev, "dias_apos_evento": lag, "c4_sem_evento_publico": ev is None})
        row["nivel3"] = bool(row["c1_proprio_do_item"] and row["c2_alta_conjunta"] and row["c3_reversao"] and row["c4_sem_evento_publico"])
        # padrão inverso (queda e recuperação), apenas descritivo
        pre = s[s["date"] < start]["price_median"].iloc[-1] if len(s[s["date"] < start]) else np.nan
        low = s[(s["date"] >= start) & (s["date"] <= end)]["price_median"].min()
        low_day = s[(s["date"] >= start) & (s["date"] <= end)].set_index("date")["price_median"].idxmin()
        aft = s[(s["date"] > low_day) & (s["date"] <= low_day + pd.Timedelta(days=7))]["price_median"]
        row["queda_e_recuperacao"] = bool(row["z_ret_min"] <= -THRESHOLD and len(aft) and aft.max() >= pre - 0.5 * (pre - low))
        rows.append(row)
ep = pd.DataFrame(rows).sort_values(["inicio", "item"]).reset_index(drop=True)
ep.to_csv(OUT / "nivel3_episodios.csv", index=False)

# Sensibilidade da regra: limiar de volume e prazo de reversão
sens = []
for vol_thr in [2.0, 3.0, 4.0]:
    for rev_days in [3, 7, 14]:
        n = 0
        for _, r in ep.iterrows():
            s = df[df["item"] == r["item"]]
            v = evaluate(s, pd.Timestamp(r["inicio"]), pd.Timestamp(r["fim"]), vol_thr, rev_days)
            n += bool(v["c1_proprio_do_item"] and v["c2_alta_conjunta"] and v["c3_reversao"] and r["c4_sem_evento_publico"])
        sens.append({"limiar_volume": vol_thr, "prazo_reversao_dias": rev_days, "episodios_nivel3": n})
pd.DataFrame(sens).to_csv(OUT / "nivel3_sensibilidade.csv", index=False)

# Calendário × índice de mercado (D0 e D+1, porque o dia do dataset pode estar em UTC)
cal_rows = []
for _, r in cal.iterrows():
    w = idx.loc[r["data"]: r["data"] + pd.Timedelta(days=1)]
    worst = w.loc[w["z_retorno_mercado"].abs().idxmax()]
    cal_rows.append({
        "data": r["data"].date(), "tipo": r["tipo"], "evento": r["evento"],
        "retorno_mercado_d0_%": round(100 * (np.exp(idx.loc[r["data"], "retorno_mercado"]) - 1), 1),
        "retorno_mercado_d1_%": round(100 * (np.exp(w["retorno_mercado"].iloc[-1]) - 1), 1),
        "maior_escore_d0_d1": round(float(worst["z_retorno_mercado"]), 1),
        "amplitude_max_d0_d7_%": round(float(breadth.loc[r["data"]: r["data"] + pd.Timedelta(days=EVENT_WINDOW)].max()), 1),
        "choque_no_indice_d0_d1": bool(w["z_retorno_mercado"].abs().max() >= THRESHOLD),
    })
calendar = pd.DataFrame(cal_rows)
calendar.to_csv(OUT / "calendario_eventos_vs_indice.csv", index=False)

# Taxa-base: que fração dos dias da janela cai em até 7 dias após algum evento do calendário?
days = idx.index
covered = pd.Series([last_event(d)[0] is not None for d in days], index=days)
shock = idx["z_retorno_mercado"].abs() >= THRESHOLD
common = breadth > BREADTH_LIMIT
resumo = {
    "episodios": len(ep),
    "c1_proprio_do_item": int(ep["c1_proprio_do_item"].sum()),
    "c2_alta_conjunta": int(ep["c2_alta_conjunta"].sum()),
    "c3_reversao": int(ep["c3_reversao"].sum()),
    "c4_sem_evento_publico": int(ep["c4_sem_evento_publico"].sum()),
    "c1_e_c4": int((ep["c1_proprio_do_item"] & ep["c4_sem_evento_publico"]).sum()),
    "c1_e_c2": int((ep["c1_proprio_do_item"] & ep["c2_alta_conjunta"]).sum()),
    "nivel3": int(ep["nivel3"].sum()),
    "queda_e_recuperacao_proprias": int((ep["queda_e_recuperacao"] & ep["c1_proprio_do_item"]).sum()),
    "sensibilidade_min_max_nivel3": [int(min(s["episodios_nivel3"] for s in sens)), int(max(s["episodios_nivel3"] for s in sens))],
    "taxa_base": {
        "eventos_no_calendario": len(cal),
        "pct_dias_da_janela_ate_7_dias_apos_evento": round(100 * covered.mean(), 1),
        "episodios_iniciados_ate_7_dias_apos_evento": int((~ep["c4_sem_evento_publico"]).sum()),
        "pct_episodios": round(100 * (~ep["c4_sem_evento_publico"]).mean(), 1),
        "dias_de_choque_do_indice": int(shock.sum()),
        "dias_de_choque_ate_7_dias_apos_evento": int((shock & covered).sum()),
        "dias_de_choque_comum_amplitude": int(common.sum()),
        "dias_de_choque_comum_ate_7_dias_apos_evento": int((common & covered).sum()),
    },
    "promocoes": {
        "total": int((calendar["tipo"] == "promoção").sum()),
        "com_retorno_negativo_em_d1": int(((calendar["tipo"] == "promoção") & (calendar["retorno_mercado_d1_%"] < 0)).sum()),
        "com_choque_no_indice_d0_d1": int(((calendar["tipo"] == "promoção") & calendar["choque_no_indice_d0_d1"]).sum()),
    },
}
(OUT / "resumo_nivel3.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
pd.set_option("display.width", 250)
print(json.dumps(resumo, ensure_ascii=False, indent=2, default=str))
print(calendar.to_string(index=False))
print(pd.DataFrame(sens).pivot(index="limiar_volume", columns="prazo_reversao_dias", values="episodios_nivel3"))
print(ep[ep["c1_proprio_do_item"] & ep["c4_sem_evento_publico"]][["item","inicio","fim","z_ret_max","z_ret_min","z_vol_max","alta_ate_pico_%","c2_alta_conjunta","c3_reversao"]].to_string(index=False))
