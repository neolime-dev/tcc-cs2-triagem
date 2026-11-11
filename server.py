#!/usr/bin/env python3
"""Terminal de triagem CS2 — servidor leve (biblioteca padrão + pandas/sklearn/ruptures já no .venv).

Reproduz as fórmulas de gerar_tabelas_cesta.py (seções 5 e 15 a 20) e de robustez_nivel3.py (episódios,
regra e calendário). O índice de mercado e as anomalias sintéticas vêm pré-calculados (output/*.csv).
Com os parâmetros padrão — limiar 4,0, penalização 10, contaminação 0,02, amplitude 10%, limiar de volume 4
e reversão em 7 dias — os números batem com o TCC.

Uso (dentro de prototipo/):   .venv/bin/python server.py   →   http://127.0.0.1:8765
Opções:  --port 8765  --host 127.0.0.1  --open (abre o navegador quando estiver pronto)
"""
import argparse
import hashlib
import socket
import sys
import webbrowser
import json
import math
import os
import threading
import time
from functools import lru_cache
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd
import ruptures as rpt
from joblib import Parallel, delayed
from sklearn.ensemble import IsolationForest

HERE = Path(__file__).resolve().parent
DATA = HERE / "output" / "cesta_2024_2026.csv"
WEB = HERE / "web"
CACHE_DIR = HERE / ".cache"
# Processos/threads paralelos. Na nuvem (Render grátis, 512 MB) o Dockerfile define N_JOBS=1.
N_JOBS = int(os.environ.get("N_JOBS", os.cpu_count() or 1))

# --- Constantes idênticas a app.py / gerar_tabelas_cesta.py ---
SELECTED_ITEMS = [
    "AK-47 | Asiimov (Field-Tested)",
    "AWP | Hyper Beast (Field-Tested)",
    "M4A1-S | Printstream (Field-Tested)",
    "Desert Eagle | Printstream (Field-Tested)",
    "Glock-18 | Water Elemental (Field-Tested)",
]
EVENT_DATE = pd.Timestamp("2025-10-22")
FEATURES = ["z_return", "z_volume", "z_volume_change", "z_rolling_vol"]
PADRAO = {"limiar": 4.0, "penalidade": 10, "contaminacao": 0.02, "amplitude": 10.0, "volz": 4.0, "reversao": 7}
JANELA_EVENTO = 7          # dias entre o evento público e o início do episódio
GRADE_VOLZ = [2.0, 3.0, 4.0]   # sensibilidade da regra: limiar de volume
GRADE_REV = [3, 7, 14]         # sensibilidade da regra: prazo de reversão (dias)
# Universo do índice de mercado (calculado por robustez_mercado.py sobre o dataset completo).
UNIVERSO = {"skins": 1412, "taxa_candidatos": 2.35, "criterio": "≥ 880 dias e volume mediano ≥ 30"}
GRADE_MAD = [3.0, 3.5, 4.0, 4.5, 5.0, 6.0]
GRADE_PELT = [5, 8, 10, 12, 15, 20]
GRADE_IF = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]
PELT_SLIDER = list(range(5, 21))  # todas as penalizações do slider (5 a 20, passo 1)
ORDEM_SKINS = [s.replace(" (Field-Tested)", "") for s in SELECTED_ITEMS]
NOME_COMPLETO = dict(zip(ORDEM_SKINS, SELECTED_ITEMS))
RARIDADE = {  # raridade oficial no CS2
    "AK-47 | Asiimov": ("Covert", "#eb4b4b"),
    "AWP | Hyper Beast": ("Covert", "#eb4b4b"),
    "M4A1-S | Printstream": ("Covert", "#eb4b4b"),
    "Desert Eagle | Printstream": ("Covert", "#eb4b4b"),
    "Glock-18 | Water Elemental": ("Classified", "#d32ce6"),
}
# Calendário de eventos públicos do TCC: 9 promoções sazonais da Steam e 5 atualizações oficiais.
# Atualizações: data UTC da publicação no canal oficial de notícias do jogo (todas entre 22h58 e 23h50 UTC).
CALENDARIO = [
    ("2024-03-14", "promoção", "Promoção de primavera 2024", "Spring Sale"),
    ("2024-06-27", "promoção", "Promoção de verão 2024", "Summer Sale"),
    ("2024-10-02", "atualização", "Atualização The Armory", "The Armory"),
    ("2024-11-27", "promoção", "Promoção de outono 2024", "Autumn Sale"),
    ("2024-12-19", "promoção", "Promoção de inverno 2024", "Winter Sale"),
    ("2025-01-28", "atualização", "Atualização Premier Season Two", "Premier S2"),
    ("2025-03-13", "promoção", "Promoção de primavera 2025", "Spring Sale"),
    ("2025-03-31", "atualização", "Atualização Spring Forward", "Spring Forward"),
    ("2025-06-26", "promoção", "Promoção de verão 2025", "Summer Sale"),
    ("2025-09-29", "promoção", "Promoção de outono 2025", "Autumn Sale"),
    ("2025-10-22", "atualização", "Atualização do Trade Up Contract", "Trade Up"),
    ("2025-11-12", "atualização", "Atualização Budapest Major Hub", "Major Hub"),
    ("2025-12-18", "promoção", "Promoção de inverno 2025", "Winter Sale"),
    ("2026-03-19", "promoção", "Promoção de primavera 2026", "Spring Sale"),
]
CAL_TS = [(pd.Timestamp(d), tipo, nome, rotulo) for d, tipo, nome, rotulo in CALENDARIO]
ATUALIZACOES = [{"date": d, "rotulo": rotulo, "texto": f"{nome} ({d[8:]}/{d[5:7]}/{d[:4]})"}
                for d, tipo, nome, rotulo in CALENDARIO if tipo == "atualização"]


# =============================================================================
# Cálculo — mesma lógica e mesmas fórmulas de app.py
# =============================================================================
def robust_z(series):
    median = series.median(skipna=True)
    mad = (series - median).abs().median(skipna=True)
    if not np.isfinite(mad) or mad == 0:
        return pd.Series(np.nan, index=series.index)
    return (series - median) / (1.4826 * mad)


def preparar_base() -> pd.DataFrame:
    """Seções 5 e 15: variáveis derivadas e escores robustos (não dependem dos parâmetros)."""
    df = pd.read_csv(DATA, usecols=["market_hash_name", "date", "price_median", "volume"])
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
    df["rolling_vol_30"] = df.groupby("market_hash_name")["log_return"].transform(
        lambda s: s.rolling(30, min_periods=15).std()
    )
    for column, output in [
        ("log_return", "z_return"),
        ("log_volume", "z_volume"),
        ("log_volume_change", "z_volume_change"),
        ("rolling_vol_30", "z_rolling_vol"),
    ]:
        df[output] = df.groupby("market_hash_name")[column].transform(robust_z)
    df["robust_score"] = df[["z_return", "z_volume"]].abs().max(axis=1)
    return df


def marcar_mad(df: pd.DataFrame, limiar: float) -> pd.DataFrame:
    df = df.copy()
    df["return_candidate"] = df["z_return"].abs() >= limiar
    df["volume_candidate"] = df["z_volume"].abs() >= limiar
    df["robust_candidate"] = df["return_candidate"] | df["volume_candidate"]
    return df


def sensibilidade_mad(df: pd.DataFrame, limiares) -> list:
    base_score = df[["z_return", "z_volume"]].abs().max(axis=1)
    return [{"limiar": float(t), "eventos": int((base_score >= t).sum()),
             "percentual": float(100 * (base_score >= t).mean())} for t in limiares]


def _pelt(signal: np.ndarray, penalidade: float) -> list:
    """Seção 17: PELT com custo RBF, segmento mínimo de 14 dias."""
    breaks = rpt.Pelt(model="rbf", min_size=14, jump=1).fit(signal).predict(pen=penalidade)
    return [int(b) for b in breaks if b < len(signal)]


def _sinal_pelt(df: pd.DataFrame, short_name: str):
    f = df[df["market_hash_name"] == NOME_COMPLETO[short_name]].copy().reset_index(drop=True)
    signal = np.column_stack([
        (f["log_price"] - f["log_price"].mean()) / f["log_price"].std(),
        (f["log_volume"] - f["log_volume"].mean()) / f["log_volume"].std(),
    ])
    return f, signal


def ajustar_isolation_forest(df: pd.DataFrame, contaminacao: float) -> pd.DataFrame:
    """Seção 18: 300 árvores, random_state=42, mesmas features e mesma ordem de linhas."""
    model_data = df.dropna(subset=FEATURES)[["market_hash_name", "date", "short_name", *FEATURES]].copy()
    iso = IsolationForest(n_estimators=300, contamination=contaminacao, random_state=42, n_jobs=N_JOBS)
    model_data["if_label"] = iso.fit_predict(model_data[FEATURES])
    model_data["if_score"] = -iso.decision_function(model_data[FEATURES])
    model_data["if_candidate"] = model_data["if_label"] == -1
    return model_data


def combinar_if_mad(model_data: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    md = model_data.merge(df[["market_hash_name", "date", "robust_candidate"]],
                          on=["market_hash_name", "date"], how="left")
    md["both"] = md["if_candidate"] & md["robust_candidate"]
    return md


def ranking_integrado(df: pd.DataFrame, model_data: pd.DataFrame, n: int = 25) -> pd.DataFrame:
    """Seção 19: média dos postos percentuais do escore robusto e do escore do IF."""
    merged = df.merge(
        model_data[["market_hash_name", "date", "if_score", "if_candidate"]],
        on=["market_hash_name", "date"], how="left",
    )
    merged["if_score_rank"] = merged["if_score"].rank(pct=True)
    merged["robust_score_rank"] = merged["robust_score"].rank(pct=True)
    merged["integrated_rank"] = merged[["if_score_rank", "robust_score_rank"]].mean(axis=1, skipna=False)
    return merged.nlargest(n, "integrated_rank")[[
        "date", "short_name", "price_median", "volume", "log_return", "z_return", "z_volume", "robust_score",
        "if_score", "robust_candidate", "if_candidate", "if_score_rank", "robust_score_rank", "integrated_rank",
    ]]


def estudo_evento(df: pd.DataFrame):
    """Seção 20: janela de ±30 dias do Trade Up com base em D−1 e volume mediana × mediana."""
    window = df[(df["date"] >= EVENT_DATE - pd.Timedelta(days=30)) & (df["date"] <= EVENT_DATE + pd.Timedelta(days=30))]
    event_rows, curvas = [], {}
    for name, g in window.groupby("short_name"):
        g = g.sort_values("date").copy()
        before = g[g["date"] < EVENT_DATE]
        after = g[g["date"] >= EVENT_DATE]
        if before.empty or after.empty:
            continue
        base_row = before.iloc[-1]
        base = base_row["price_median"]

        def price_at(days, g=g):
            return g[g["date"] <= EVENT_DATE + pd.Timedelta(days=days)].iloc[-1]["price_median"]

        g["event_index"] = 100 * g["price_median"] / base
        curvas[name] = {"date": g["date"].dt.strftime("%Y-%m-%d").tolist(),
                        "index": g["event_index"].round(4).tolist(), "price": g["price_median"].tolist()}
        event_rows.append({
            "item": name,
            "data_base": base_row["date"].strftime("%Y-%m-%d"),
            "preco_base": base,
            "retorno_d0": 100 * (price_at(0) / base - 1),
            "retorno_7d": 100 * (price_at(7) / base - 1),
            "retorno_30d": 100 * (price_at(30) / base - 1),
            "preco_d0": price_at(0), "preco_7d": price_at(7), "preco_30d": price_at(30),
            "volume_mediano_antes": before["volume"].median(),
            "volume_mediano_depois": after["volume"].median(),
            "razao_volume_mediano": after["volume"].median() / before["volume"].median(),
            "volume_maximo_antes": before["volume"].max(),
            "volume_maximo_depois": after["volume"].max(),
        })
    ordem = {s: i for i, s in enumerate(ORDEM_SKINS)}
    event_rows.sort(key=lambda r: ordem.get(r["item"], 99))
    return event_rows, curvas


def _achar(nome: str) -> Path:
    """Arquivos pré-calculados: output/ (deploy) ou output/robustez/ (pasta do protótipo)."""
    for p in (HERE / "output" / nome, HERE / "output" / "robustez" / nome):
        if p.exists():
            return p
    raise SystemExit(f"Arquivo não encontrado: output/{nome} (gere com robustez_mercado.py / robustez_banca.py)")


def anexar_mercado(df: pd.DataFrame, mercado: pd.DataFrame) -> pd.DataFrame:
    """Índice de mercado (mediana de 1.412 skins) e baseline ajustado: escore sobre item − mercado."""
    cols = ["retorno_mercado", "volume_mercado", "pct_universo_marcado"]
    df = df.merge(mercado[cols].reset_index(), on="date", how="left")
    df["adj_return"] = df["log_return"] - df["retorno_mercado"]
    df["adj_volume"] = df["log_volume"] - df["volume_mercado"]
    df["z_return_adj"] = df.groupby("market_hash_name")["adj_return"].transform(robust_z)
    df["z_volume_adj"] = df.groupby("market_hash_name")["adj_volume"].transform(robust_z)
    df["robust_score_adj"] = df[["z_return_adj", "z_volume_adj"]].abs().max(axis=1)
    return df


def listar_episodios(df: pd.DataFrame, coluna: str) -> list:
    """Dias marcados do mesmo item separados por até dois dias formam um episódio."""
    eps = []
    for s in ORDEM_SKINS:
        c = df[(df["short_name"] == s) & df[coluna]].sort_values("date")
        if c.empty:
            continue
        bloco = (c["date"].diff().dt.days > 2).cumsum()
        for _, e in c.groupby(bloco):
            eps.append((s, e["date"].min(), e["date"].max(), len(e)))
    return eps


def evento_associado(inicio: pd.Timestamp, fim: pd.Timestamp):
    """Evento do calendário entre JANELA_EVENTO dias antes do início e o fim do episódio."""
    hits = [c for c in CAL_TS if inicio - pd.Timedelta(days=JANELA_EVENTO) <= c[0] <= fim]
    return hits[-1] if hits else None


def classificar_evento(data: pd.Timestamp, short_name: str, amplitude: float, limite: float) -> dict:
    """Classificação das prioridades: abrangência (amplitude) + contexto registrado no TCC."""
    comum = amplitude is not None and amplitude > limite
    grupo = "Choque comum" if comum else "Movimento próprio do item"
    amp_txt = "—" if amplitude is None else f"{amplitude:.1f}".replace(".", ",") + "%"
    prefixo = f"{grupo} (amplitude de {amp_txt})"
    if pd.Timestamp("2025-10-22") <= data <= pd.Timestamp("2025-11-02"):
        return {"grupo": grupo, "badge": "tradeup", "rotulo": "Trade Up", "amp_txt": amp_txt,
                "texto": "Choque comum; associado a contexto legítimo conhecido: atualização oficial de 22/10/2025 (Trade Up)"}
    if data == pd.Timestamp("2024-10-03"):
        return {"grupo": grupo, "badge": "armory", "rotulo": "The Armory", "amp_txt": amp_txt,
                "texto": "Choque comum; associado a contexto legítimo conhecido: atualização oficial The Armory, "
                         "de 02/10/2024 (publicada às 23h50 UTC; efeito no registro de 03/10)"}
    if short_name.startswith("M4A1-S") and data in (pd.Timestamp("2025-06-24"), pd.Timestamp("2025-06-25")):
        return {"grupo": grupo, "badge": "liquidez", "rotulo": "Baixa liquidez", "amp_txt": amp_txt,
                "texto": "Movimento próprio do item; sem explicação observada; hipótese de problema de dado ou venda isolada"}
    if short_name.startswith("AK-47") and data == pd.Timestamp("2025-02-01"):
        return {"grupo": grupo, "badge": "semexp", "rotulo": "Causa não confirmada", "amp_txt": amp_txt,
                "texto": "Choque comum; coincide com a atualização de 28/01/2025 e com relatos de falha no mercado, "
                         "sem confirmação oficial"}
    ev = evento_associado(data, data)
    extra = f"; até {JANELA_EVENTO} dias após {ev[2]}" if ev else "; sem evento público associado"
    return {"grupo": grupo, "badge": "nc", "rotulo": "Choque comum" if comum else "Próprio do item", "amp_txt": amp_txt,
            "texto": prefixo + extra + " (fora da tabela do TCC)"}


def _num(v, casas=None):
    """Converte para tipo JSON (NaN → None)."""
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    if not math.isfinite(f):
        return None
    return round(f, casas) if casas is not None else f


def _lista(serie, casas=None):
    return [_num(v, casas) for v in serie]


class Estado:
    def __init__(self):
        t0 = time.time()
        # Identifica o CSV pelo conteúdo (não pela data): o cache feito no build vale na nuvem.
        self.chave = hashlib.sha256(DATA.read_bytes()).hexdigest()
        self.mercado = pd.read_csv(_achar("mercado_indice_diario.csv"), parse_dates=["date"]).set_index("date")
        self.base = anexar_mercado(preparar_base(), self.mercado)
        self.series = {s: g.sort_values("date").reset_index(drop=True) for s, g in self.base.groupby("short_name")}
        self.mercado_json, self.calendario, self.pct_dias_evento = self._mercado_estatico()
        self.sinteticas = self._sinteticas()
        print(f"[base] {len(self.base):,} observações carregadas em {time.time() - t0:.1f}s")

        t1 = time.time()
        self.if_models = {c: ajustar_isolation_forest(self.base, c) for c in GRADE_IF}
        print(f"[IF]   {len(GRADE_IF)} contaminações ajustadas em {time.time() - t1:.1f}s")

        t2 = time.time()
        self.pelt = self._carregar_pelt()
        print(f"[PELT] {len(ORDEM_SKINS)} skins × {len(PELT_SLIDER)} penalizações prontas em {time.time() - t2:.1f}s")

        self.evento, self.curvas_evento = estudo_evento(self.base)
        self.sens_mad_fina = sensibilidade_mad(self.base, np.round(np.arange(3.0, 6.0001, 0.1), 1))
        self.lock = threading.Lock()
        print(f"[ok]   pronto em {time.time() - t0:.1f}s")

    def _mercado_estatico(self):
        """Partes que não dependem dos parâmetros: série do índice, calendário × índice e taxa-base."""
        m = self.mercado
        ret = m["retorno_mercado"].fillna(0.0)
        serie = {
            "date": [d.strftime("%Y-%m-%d") for d in m.index],
            "nivel": _lista(100 * np.exp(ret.cumsum()), 2),
            "ret": _lista(100 * (np.exp(m["retorno_mercado"]) - 1), 2),
            "z": _lista(m["z_retorno_mercado"], 1),
            "amp": _lista(m["pct_universo_marcado"], 1),
        }
        cal = []
        for data, tipo, nome, rotulo in CAL_TS:
            w = m.loc[data: data + pd.Timedelta(days=1)]
            pior = w.loc[w["z_retorno_mercado"].abs().idxmax()]
            cal.append({
                "date": data.strftime("%Y-%m-%d"), "tipo": tipo, "evento": nome, "rotulo": rotulo,
                "d0": _num(100 * (np.exp(m.loc[data, "retorno_mercado"]) - 1), 1),
                "d1": _num(100 * (np.exp(w["retorno_mercado"].iloc[-1]) - 1), 1),
                "escore": _num(pior["z_retorno_mercado"], 1),
                "amp_max": _num(m.loc[data: data + pd.Timedelta(days=JANELA_EVENTO), "pct_universo_marcado"].max(), 1),
                "choque": bool(w["z_retorno_mercado"].abs().max() >= PADRAO["limiar"]),
            })
        cobertos = [any(0 <= (d - c[0]).days <= JANELA_EVENTO for c in CAL_TS) for d in m.index]
        return serie, cal, round(100 * float(np.mean(cobertos)), 1)

    def _sinteticas(self) -> list:
        """Taxa de recuperação de picos artificiais (pré-calculada por robustez_banca.py)."""
        t = pd.read_csv(_achar("anomalias_sinteticas_recuperacao.csv"))
        linhas = []
        for (tipo, dur, preco, vol), g in t.groupby(["tipo", "duracao_dias", "alta_preco_%", "mult_volume"], sort=False):
            por = {r["metodo"]: r for _, r in g.iterrows()}
            linhas.append({
                "tipo": tipo, "duracao": int(dur), "preco": _num(preco), "volume": _num(vol),
                "global": _num(por["global"]["recuperacao_geral_%"], 1),
                "movel90": _num(por["movel90"]["recuperacao_geral_%"], 1),
                "if": _num(por["if"]["recuperacao_geral_%"], 1),
                "por_skin": {s: _num(por["global"][s], 0) for s in ORDEM_SKINS},
            })
        return linhas

    def _episodios(self, df: pd.DataFrame, limiar: float, amplitude: float, volz: float, reversao: int):
        """Episódios do baseline e as quatro condições da regra de possível sinal de manipulação."""
        amp = self.mercado["pct_universo_marcado"]
        prazos = sorted(set(GRADE_REV) | {reversao})
        lista = []
        for s, ini, fim, n in listar_episodios(df, "robust_candidate"):
            g = self.series[s]
            ep = g[(g["date"] >= ini) & (g["date"] <= fim)]
            antes = g[g["date"] < ini]
            p_pre = antes["price_median"].iloc[-1] if len(antes) else float("nan")
            pico = ep.loc[ep["price_median"].idxmax()]
            alta = pico["price_median"] / p_pre - 1

            def reverte(dias):
                depois = g[(g["date"] > pico["date"]) & (g["date"] <= pico["date"] + pd.Timedelta(days=dias))]
                return bool(alta > 0 and len(depois)
                            and depois["price_median"].min() <= p_pre + 0.5 * (pico["price_median"] - p_pre))

            zr, zv = ep["z_return"].max(), ep["z_volume"].max()
            amp_max = float(amp.loc[ini:fim].max())
            ev = evento_associado(ini, fim)
            lista.append({
                "skin": s, "inicio": ini.strftime("%Y-%m-%d"), "fim": fim.strftime("%Y-%m-%d"), "dias": int(n),
                "amp_max": _num(amp_max, 1), "z_ret_max": _num(zr, 1), "z_vol_max": _num(zv, 1),
                "alta": _num(100 * alta, 1), "evento": ev[2] if ev else None,
                "c1": bool(amp_max <= amplitude), "c4": ev is None,
                "_zr": float(zr) if zr == zr else float("-inf"), "_zv": float(zv) if zv == zv else float("-inf"),
                "_rev": {p: reverte(p) for p in prazos},
            })
        for e in lista:
            e["c2"] = bool(e["_zr"] >= limiar and e["_zv"] >= volz)
            e["c3"] = e["_rev"][reversao]
            e["nivel3"] = bool(e["c1"] and e["c2"] and e["c3"] and e["c4"])
        sens = [{"volz": v, "reversao": p,
                 "n": sum(1 for e in lista if e["c1"] and e["c4"] and e["_zr"] >= limiar and e["_zv"] >= v and e["_rev"][p])}
                for v in GRADE_VOLZ for p in GRADE_REV]
        resumo = {
            "total": len(lista), "item_dias": int(df["robust_candidate"].sum()),
            "c1": sum(e["c1"] for e in lista), "c2": sum(e["c2"] for e in lista),
            "c3": sum(e["c3"] for e in lista), "c4": sum(e["c4"] for e in lista),
            "c1c4": sum(e["c1"] and e["c4"] for e in lista), "c1c2": sum(e["c1"] and e["c2"] for e in lista),
            "nivel3": sum(e["nivel3"] for e in lista), "associados": sum(not e["c4"] for e in lista),
        }
        lista.sort(key=lambda e: (e["inicio"], e["skin"]))
        for e in lista:
            for k in ("_zr", "_zv", "_rev"):
                e.pop(k)
        return lista, resumo, sens

    def _ajustado(self, df: pd.DataFrame, limiar: float) -> list:
        """Baseline original × baseline ajustado ao mercado, por skin e no total."""
        d = df.copy()
        d["adj_candidate"] = d["robust_score_adj"] >= limiar
        ep_o = pd.Series([e[0] for e in listar_episodios(d, "robust_candidate")]).value_counts()
        ep_a = pd.Series([e[0] for e in listar_episodios(d, "adj_candidate")]).value_counts()
        linhas = []
        for s in ORDEM_SKINS:
            g = d[d["short_name"] == s]
            linhas.append({"skin": s, "original": int(g["robust_candidate"].sum()), "ajustado": int(g["adj_candidate"].sum()),
                           "comum": int((g["robust_candidate"] & g["adj_candidate"]).sum()),
                           "ep_original": int(ep_o.get(s, 0)), "ep_ajustado": int(ep_a.get(s, 0))})
        linhas.append({"skin": "Total", **{k: sum(r[k] for r in linhas) for k in
                                           ("original", "ajustado", "comum", "ep_original", "ep_ajustado")}})
        return linhas

    def _carregar_pelt(self) -> dict:
        """PELT de todas as skins × penalizações do slider; cache em disco invalidado pelo hash do CSV."""
        CACHE_DIR.mkdir(exist_ok=True)
        arquivo = CACHE_DIR / "pelt.json"
        if arquivo.exists():
            try:
                salvo = json.loads(arquivo.read_text())
                if salvo.get("chave") == self.chave:
                    return {(s, int(p)): v for s, d in salvo["breaks"].items() for p, v in d.items()}
            except Exception:
                pass
        tarefas, frames = [], {}
        for s in ORDEM_SKINS:
            f, signal = _sinal_pelt(self.base, s)
            frames[s] = f
            tarefas += [(s, p, signal) for p in PELT_SLIDER]
        try:
            res = Parallel(n_jobs=min(len(tarefas), N_JOBS))(delayed(_pelt)(sig, p) for _, p, sig in tarefas)
        except Exception:
            res = [_pelt(sig, p) for _, p, sig in tarefas]
        pelt = {}
        for (s, p, _), idx in zip(tarefas, res):
            pelt[(s, p)] = [frames[s].iloc[b]["date"].strftime("%Y-%m-%d") for b in idx]
        dump = {"chave": self.chave, "breaks": {}}
        for (s, p), v in pelt.items():
            dump["breaks"].setdefault(s, {})[str(p)] = v
        arquivo.write_text(json.dumps(dump))
        return pelt

    # ------------------------------------------------------------------
    @lru_cache(maxsize=256)
    def payload(self, skin: str, limiar: float, penalidade: int, contaminacao: float,
                amplitude: float = PADRAO["amplitude"], volz: float = PADRAO["volz"],
                reversao: int = PADRAO["reversao"]) -> bytes:
        base = self.base
        df = marcar_mad(base, limiar)
        md = combinar_if_mad(self.if_models[contaminacao], df)
        df_ref = marcar_mad(base, PADRAO["limiar"])
        md_ref = combinar_if_mad(self.if_models[PADRAO["contaminacao"]], df_ref)
        oficial = (limiar == PADRAO["limiar"] and penalidade == PADRAO["penalidade"]
                   and contaminacao == PADRAO["contaminacao"] and amplitude == PADRAO["amplitude"]
                   and volz == PADRAO["volz"] and reversao == PADRAO["reversao"])
        amp_dia = self.mercado["pct_universo_marcado"]

        # --- cards das skins ---
        cards = []
        for s in ORDEM_SKINS:
            g = df[df["short_name"] == s]
            m = md[md["short_name"] == s]
            rar, cor = RARIDADE[s]
            ultimo = g.sort_values("date").iloc[-1]
            cards.append({
                "skin": s, "raridade": rar, "cor": cor,
                "preco_mediano": _num(g["price_median"].median()),
                "volume_mediano": _num(g["volume"].median()),
                "alertas_mad": int(g["robust_candidate"].sum()),
                "alertas_if": int(m["if_candidate"].sum()),
                "ambos": int(m["both"].sum()),
                "ultimo_preco": _num(ultimo["price_median"]),
                "ultima_data": ultimo["date"].strftime("%Y-%m-%d"),
                "spark": _lista(g.sort_values("date")["price_median"].iloc[::7], 2),
            })

        # --- KPIs globais (com referência do TCC) ---
        kpis = {
            "observacoes": int(len(df)), "skins": int(df["short_name"].nunique()),
            "inicio": df["date"].min().strftime("%Y-%m-%d"), "fim": df["date"].max().strftime("%Y-%m-%d"),
            "mad": int(df["robust_candidate"].sum()), "mad_ref": int(df_ref["robust_candidate"].sum()),
            "if": int(md["if_candidate"].sum()), "if_ref": int(md_ref["if_candidate"].sum()),
            "ambos": int(md["both"].sum()), "ambos_ref": int(md_ref["both"].sum()),
            "avaliadas_if": int(len(md)),
        }

        # --- série da skin selecionada ---
        serie = df[df["short_name"] == skin].sort_values("date")
        sk_md = md[md["short_name"] == skin][["date", "if_score", "if_candidate"]]
        serie = serie.merge(sk_md, on="date", how="left")
        serie_json = {
            "date": serie["date"].dt.strftime("%Y-%m-%d").tolist(),
            "price": _lista(serie["price_median"], 4),
            "volume": _lista(serie["volume"]),
            "log_return": _lista(serie["log_return"], 6),
            "z_return": _lista(serie["z_return"], 4),
            "z_volume": _lista(serie["z_volume"], 4),
            "robust_score": _lista(serie["robust_score"], 4),
            "if_score": _lista(serie["if_score"], 5),
            "ret_cand": [bool(v) for v in serie["return_candidate"]],
            "vol_cand": [bool(v) for v in serie["volume_candidate"]],
            "if_cand": [bool(v) if v == v else False for v in serie["if_candidate"]],
        }

        # --- ranking integrado Top 25 ---
        top = ranking_integrado(df, md)
        mediana_vol = df.groupby("short_name")["volume"].median()
        top_json = []
        for pos, (_, r) in enumerate(top.iterrows(), start=1):
            amp_evento = _num(amp_dia.get(r["date"]), 1)
            cls = classificar_evento(r["date"], r["short_name"], amp_evento, amplitude)
            g = base[base["short_name"] == r["short_name"]].sort_values("date")
            anteriores = g[g["date"] < r["date"]]
            preco_ant = anteriores.iloc[-1]["price_median"] if len(anteriores) else None
            quebras = self.pelt[(r["short_name"], penalidade)]
            dist_quebra = min((abs((pd.Timestamp(q) - r["date"]).days), q) for q in quebras) if quebras else None
            dist_valve = min((abs((pd.Timestamp(a["date"]) - r["date"]).days), a["rotulo"]) for a in ATUALIZACOES)
            top_json.append({
                "pos": pos, "date": r["date"].strftime("%Y-%m-%d"), "skin": r["short_name"],
                "price": _num(r["price_median"]), "volume": _num(r["volume"]),
                "preco_anterior": _num(preco_ant), "volume_mediano_skin": _num(mediana_vol[r["short_name"]]),
                "log_return": _num(r["log_return"]), "z_return": _num(r["z_return"]), "z_volume": _num(r["z_volume"]),
                "robust_score": _num(r["robust_score"]), "if_score": _num(r["if_score"]),
                "robust_rank": _num(r["robust_score_rank"]), "if_rank": _num(r["if_score_rank"]),
                "integrated_rank": _num(r["integrated_rank"]),
                "mad": bool(r["robust_candidate"]), "if": bool(r["if_candidate"]) if r["if_candidate"] == r["if_candidate"] else False,
                "quebra_proxima": {"dias": dist_quebra[0], "data": dist_quebra[1]} if dist_quebra else None,
                "valve_proxima": {"dias": dist_valve[0], "rotulo": dist_valve[1]},
                "amplitude": amp_evento, "comum": cls["grupo"] == "Choque comum",
                **cls,
            })

        # --- agrupamento por período (como na tabela do TCC) ---
        grupos = {}
        for e in top_json:
            grupos.setdefault(e["texto"], []).append(e)
        grupos_json = []
        for texto, itens in grupos.items():
            datas = sorted(e["date"] for e in itens)
            armas = pd.Series([e["skin"].split(" | ")[0] for e in itens]).value_counts()
            armas = sorted(armas.items(), key=lambda kv: (-kv[1], kv[0]))
            grupos_json.append({"texto": texto, "badge": itens[0]["badge"], "rotulo": itens[0]["rotulo"],
                                "n": len(itens), "inicio": datas[0], "fim": datas[-1],
                                "armas": [{"arma": a, "n": int(n)} for a, n in armas]})

        # --- sensibilidade ---
        sens_if = []
        for c in GRADE_IF:
            m = combinar_if_mad(self.if_models[c], df)
            sens_if.append({"contaminacao": c, "eventos_if": int(m["if_candidate"].sum()), "ambos": int(m["both"].sum())})
        pelt_grade = [{"penalizacao": p, "datas": self.pelt[(skin, p)], "no_tcc": p in GRADE_PELT} for p in PELT_SLIDER]
        em_todas = sorted(set.intersection(*(set(self.pelt[(skin, p)]) for p in GRADE_PELT)))

        # --- mercado, episódios e regra ---
        episodios, ep_resumo, ep_sens = self._episodios(df, limiar, amplitude, volz, reversao)
        dias_comuns = int((amp_dia > amplitude).sum())
        mercado = {
            "serie": self.mercado_json, "universo": UNIVERSO, "limite": amplitude,
            "dias_comuns": dias_comuns, "dias": int(len(amp_dia)),
            "amp_mediana": _num(amp_dia.median(), 2),
            "top_comum": sum(1 for e in top_json if e["comum"]), "top_total": len(top_json),
            "calendario": self.calendario, "pct_dias_evento": self.pct_dias_evento,
            "ajustado": self._ajustado(df, limiar),
        }

        out = {
            "params": {"skin": skin, "limiar": limiar, "penalidade": penalidade, "contaminacao": contaminacao,
                       "amplitude": amplitude, "volz": volz, "reversao": reversao, "oficial": oficial},
            "mercado": mercado,
            "regra": {"episodios": episodios, "resumo": ep_resumo, "sens": ep_sens,
                      "grade_volz": GRADE_VOLZ, "grade_rev": GRADE_REV},
            "sinteticas": self.sinteticas,
            "padrao": PADRAO, "cards": cards, "kpis": kpis, "serie": serie_json,
            "pelt": self.pelt[(skin, penalidade)], "atualizacoes": ATUALIZACOES,
            "top": top_json, "grupos": grupos_json,
            "evento": {"data": EVENT_DATE.strftime("%Y-%m-%d"), "resumo": [{k: _num(v) for k, v in r.items()}
                                                                       for r in self.evento],
                       "curvas": self.curvas_evento},
            "sens": {"mad_fina": self.sens_mad_fina, "grade_mad": GRADE_MAD,
                     "mad_grade": sensibilidade_mad(base, sorted(set(GRADE_MAD) | {limiar})),
                     "pelt": pelt_grade, "pelt_em_todas": em_todas, "grade_pelt": GRADE_PELT,
                     "if": sens_if},
        }
        return json.dumps(out, ensure_ascii=False, allow_nan=False).encode("utf-8")


# =============================================================================
# HTTP
# =============================================================================
ESTADO: Estado | None = None


def _parse(qs: dict):
    def get(nome, padrao):
        v = qs.get(nome, [None])[0]
        return padrao if v in (None, "") else v

    skin = get("skin", ORDEM_SKINS[0])
    if skin not in ORDEM_SKINS:
        skin = next((s for s in ORDEM_SKINS if s.lower().startswith(skin.lower()[:4])), None)
        if skin is None:
            raise ValueError("skin desconhecida")
    limiar = round(min(max(float(get("limiar", PADRAO["limiar"])), 3.0), 6.0), 1)
    penalidade = int(round(min(max(float(get("penalidade", PADRAO["penalidade"])), 5), 20)))
    c = float(get("contaminacao", PADRAO["contaminacao"]))
    contaminacao = min(GRADE_IF, key=lambda g: abs(g - c))  # passo de 0,005 do slider
    amplitude = float(round(min(max(float(get("amplitude", PADRAO["amplitude"])), 5), 30)))
    volz = round(min(max(float(get("volz", PADRAO["volz"])), 2.0), 4.0) * 2) / 2
    reversao = int(round(min(max(float(get("reversao", PADRAO["reversao"])), 3), 14)))
    return skin, limiar, penalidade, contaminacao, amplitude, volz, reversao


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(WEB), **kw)

    def log_message(self, fmt, *args):  # log curto
        if args and isinstance(args[0], str) and "/api/" in args[0]:
            print(f"[api]  {args[0]} → {args[1]}")

    def end_headers(self):
        if not self.path.startswith("/api/"):
            self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def _json(self, corpo: bytes, status=HTTPStatus.OK):
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(corpo)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(corpo)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path.startswith("/api/") and ESTADO is None:  # ainda calculando: o front tenta de novo
            return self._json(b'{"carregando": true}', HTTPStatus.SERVICE_UNAVAILABLE)
        if url.path == "/api/state":
            try:
                args = _parse(parse_qs(url.query))
            except (ValueError, TypeError) as e:
                return self._json(json.dumps({"erro": str(e)}).encode(), HTTPStatus.BAD_REQUEST)
            with ESTADO.lock:
                corpo = ESTADO.payload(*args)
            return self._json(corpo)
        if url.path == "/api/health":
            return self._json(b'{"ok": true}')
        return super().do_GET()


def main():
    global ESTADO
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--open", action="store_true", help="abre o navegador quando o servidor estiver pronto")
    a = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    url = f"http://{a.host}:{a.port}"
    with socket.socket() as sk:
        if sk.connect_ex((a.host, a.port)) == 0:  # já está rodando: só abre o navegador
            print(f"Servidor já ativo em {url}")
            if a.open:
                webbrowser.open(url)
            return
    if not DATA.exists():
        raise SystemExit(f"Arquivo não encontrado: {DATA}")
    # Abre a porta antes de calcular (o Render exige porta aberta); a API responde 503 até ficar pronta.
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        estado = Estado()
        estado.payload(ORDEM_SKINS[0], PADRAO["limiar"], PADRAO["penalidade"], PADRAO["contaminacao"])  # aquece
        ESTADO = estado
        print(f"\n  Terminal de triagem CS2 → {url}\n  (Ctrl+C para encerrar)\n")
        if a.open:
            webbrowser.open(url)
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        srv.shutdown()


if __name__ == "__main__":
    main()
