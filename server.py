#!/usr/bin/env python3
"""Terminal de triagem CS2 — servidor leve (biblioteca padrão + pandas/sklearn/ruptures já no .venv).

Reproduz exatamente as fórmulas de app.py / gerar_tabelas_cesta.py (seções 5 e 15 a 20). Com os
parâmetros padrão — limiar MAD 4,0, penalização PELT 10 e contaminação 0,02 — os números batem com o TCC.

Uso (dentro de prototipo/):   .venv/bin/python server.py   →   http://127.0.0.1:8765
Opções:  --port 8765  --host 127.0.0.1  --open (abre o navegador quando estiver pronto)
"""
import argparse
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
PADRAO = {"limiar": 4.0, "penalidade": 10, "contaminacao": 0.02}
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
ATUALIZACOES = [
    {"date": "2024-10-02", "rotulo": "The Armory", "texto": "Atualização The Armory (02/10/2024)"},
    {"date": "2025-10-22", "rotulo": "Trade Up", "texto": "Atualização Trade Up (22/10/2025)"},
]


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


def classificar_evento(data: pd.Timestamp, short_name: str) -> dict:
    """Classificação contextual manual registrada no TCC (mesmas regras de app.py)."""
    if pd.Timestamp("2025-10-22") <= data <= pd.Timestamp("2025-11-02"):
        return {"grupo": "Atualização oficial", "badge": "tradeup", "rotulo": "Trade Up",
                "texto": "Associado a contexto legítimo conhecido: atualização oficial de 22/10/2025 (Trade Up)"}
    if data == pd.Timestamp("2024-10-03"):
        return {"grupo": "Atualização oficial", "badge": "armory", "rotulo": "The Armory",
                "texto": "Associado a contexto legítimo conhecido: atualização oficial The Armory, "
                         "de 02/10/2024 (03/10 em UTC)"}
    if short_name.startswith("M4A1-S") and data in (pd.Timestamp("2025-06-24"), pd.Timestamp("2025-06-25")):
        return {"grupo": "Sem explicação verificada", "badge": "liquidez", "rotulo": "Baixa liquidez",
                "texto": "Sem explicação observada; hipótese de problema de dado ou venda isolada"}
    if short_name.startswith("AK-47") and data == pd.Timestamp("2025-02-01"):
        return {"grupo": "Sem explicação verificada", "badge": "semexp", "rotulo": "Sem explicação",
                "texto": "Sem explicação observada; mantido para investigação"}
    return {"grupo": "Não classificado", "badge": "nc", "rotulo": "Não classificado",
            "texto": "Fora da classificação do TCC; exige investigação contextual"}


# =============================================================================
# Estado em memória (pré-carregado na inicialização)
# =============================================================================
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
        self.mtime = DATA.stat().st_mtime
        self.base = preparar_base()
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

    def _carregar_pelt(self) -> dict:
        """PELT de todas as skins × penalizações do slider; cache em disco invalidado pelo mtime do CSV."""
        CACHE_DIR.mkdir(exist_ok=True)
        arquivo = CACHE_DIR / "pelt.json"
        if arquivo.exists():
            try:
                salvo = json.loads(arquivo.read_text())
                if salvo.get("mtime") == self.mtime:
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
        dump = {"mtime": self.mtime, "breaks": {}}
        for (s, p), v in pelt.items():
            dump["breaks"].setdefault(s, {})[str(p)] = v
        arquivo.write_text(json.dumps(dump))
        return pelt

    # ------------------------------------------------------------------
    @lru_cache(maxsize=256)
    def payload(self, skin: str, limiar: float, penalidade: int, contaminacao: float) -> bytes:
        base = self.base
        df = marcar_mad(base, limiar)
        md = combinar_if_mad(self.if_models[contaminacao], df)
        df_ref = marcar_mad(base, PADRAO["limiar"])
        md_ref = combinar_if_mad(self.if_models[PADRAO["contaminacao"]], df_ref)
        oficial = (limiar == PADRAO["limiar"] and penalidade == PADRAO["penalidade"]
                   and contaminacao == PADRAO["contaminacao"])

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
            cls = classificar_evento(r["date"], r["short_name"])
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

        out = {
            "params": {"skin": skin, "limiar": limiar, "penalidade": penalidade, "contaminacao": contaminacao,
                       "oficial": oficial},
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
    return skin, limiar, penalidade, contaminacao


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(WEB), **kw)

    def log_message(self, fmt, *args):  # log curto
        if "/api/" in (args[0] if args else ""):
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
    ESTADO = Estado()
    ESTADO.payload(ORDEM_SKINS[0], PADRAO["limiar"], PADRAO["penalidade"], PADRAO["contaminacao"])  # aquece
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"\n  Terminal de triagem CS2 → {url}\n  (Ctrl+C para encerrar)\n")
    if a.open:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
