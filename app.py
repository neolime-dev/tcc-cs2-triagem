#!/usr/bin/env python3
"""Mesa de triagem interativa para a banca — TCC CS2 / Steam Community Market.

Lê output/cesta_2024_2026.csv e reproduz, com parâmetros ajustáveis, exatamente a lógica de
gerar_tabelas_cesta.py (seções 5 e 15 a 20). Com os parâmetros padrão — limiar MAD 4,0,
penalização PELT 10 e contaminação 0,02 — todos os números coincidem com as tabelas do TCC.

Uso (dentro de prototipo/):  .venv/bin/streamlit run app.py
"""
import os
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import ruptures as rpt
import streamlit as st
from joblib import Parallel, delayed
from sklearn.ensemble import IsolationForest

HERE = Path(__file__).resolve().parent
DATA = HERE / "output" / "cesta_2024_2026.csv"

# --- Constantes idênticas a gerar_tabelas_cesta.py ---
SELECTED_ITEMS = [
    "AK-47 | Asiimov (Field-Tested)",
    "AWP | Hyper Beast (Field-Tested)",
    "M4A1-S | Printstream (Field-Tested)",
    "Desert Eagle | Printstream (Field-Tested)",
    "Glock-18 | Water Elemental (Field-Tested)",
]
EVENT_DATE = pd.Timestamp("2025-10-22")
FEATURES = ["z_return", "z_volume", "z_volume_change", "z_rolling_vol"]

# --- Parâmetros do TCC (padrão dos sliders) e grades de sensibilidade do texto ---
PADRAO = {"limiar": 4.0, "penalidade": 10, "contaminacao": 0.02}
GRADE_MAD = [3.0, 3.5, 4.0, 4.5, 5.0, 6.0]
GRADE_PELT = [5, 8, 10, 12, 15, 20]
GRADE_IF = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]

ORDEM_SKINS = [s.replace(" (Field-Tested)", "") for s in SELECTED_ITEMS]
NOME_COMPLETO = dict(zip(ORDEM_SKINS, SELECTED_ITEMS))
ATUALIZACOES = pd.DataFrame({
    "date": pd.to_datetime(["2024-10-02", "2025-10-22"]),
    "rotulo": ["The Armory (02/10/2024)", "Trade Up (22/10/2025)"],
})

# --- Paleta (validada para daltonismo; cor fixa por skin, nunca por posição) ---
AZUL, AZUL_CLARO, VERMELHO, VERDE = "#2a78d6", "#86b6ef", "#e34948", "#008300"
CINZA, CINZA_TEXTO = "#52514e", "#6b6a66"
CORES_SKIN = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
LEG_CAND = "Candidato MAD (|z| ≥ limiar)"
LEG_PELT = "Quebra de regime (PELT)"
LEG_VALVE = "Atualização oficial da Valve"
MEIO_DIA = pd.Timedelta(hours=12)  # datas plotadas ao meio-dia: o fuso do navegador não muda o dia exibido


# =============================================================================
# Cálculo — mesma lógica e mesmas fórmulas de gerar_tabelas_cesta.py
# =============================================================================
def robust_z(series):
    median = series.median(skipna=True)
    mad = (series - median).abs().median(skipna=True)
    if not np.isfinite(mad) or mad == 0:
        return pd.Series(np.nan, index=series.index)
    return (series - median) / (1.4826 * mad)


@st.cache_data(show_spinner="Carregando a cesta e calculando os escores robustos…")
def preparar_base(mtime: float) -> pd.DataFrame:
    """Seções 5 e 15: variáveis derivadas e escores robustos (não dependem dos sliders)."""
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
    """Seção 15: candidatos do baseline robusto para o limiar escolhido."""
    df = df.copy()
    df["return_candidate"] = df["z_return"].abs() >= limiar
    df["volume_candidate"] = df["z_volume"].abs() >= limiar
    df["robust_candidate"] = df["return_candidate"] | df["volume_candidate"]
    return df


def contagem_mad(df: pd.DataFrame) -> pd.DataFrame:
    counts = df.groupby("short_name").agg(
        observacoes=("date", "size"),
        retorno_candidato=("return_candidate", "sum"),
        volume_candidato=("volume_candidate", "sum"),
        uniao=("robust_candidate", "sum"),
    )
    counts["percentual_uniao"] = 100 * counts["uniao"] / counts["observacoes"]
    return counts


def sensibilidade_mad(df: pd.DataFrame, limiares) -> pd.DataFrame:
    """Seção 16."""
    base_score = df[["z_return", "z_volume"]].abs().max(axis=1)
    return pd.DataFrame([
        {"limiar": t, "eventos": int((base_score >= t).sum()), "percentual": 100 * (base_score >= t).mean()}
        for t in limiares
    ])


def _pelt(signal: np.ndarray, penalidade: float) -> list:
    """Seção 17: PELT com custo RBF, segmento mínimo de 14 dias."""
    breaks = rpt.Pelt(model="rbf", min_size=14, jump=1).fit(signal).predict(pen=penalidade)
    return [b for b in breaks if b < len(signal)]


@st.cache_resource
def _memoria_pelt() -> dict:
    return {}


def obter_pelt(df: pd.DataFrame, short_name: str, penalidades, mtime: float) -> dict:
    """Pontos de mudança por penalização (datas = primeiro dia do novo segmento)."""
    f = df[df["market_hash_name"] == NOME_COMPLETO[short_name]].copy().reset_index(drop=True)
    signal = np.column_stack([
        (f["log_price"] - f["log_price"].mean()) / f["log_price"].std(),
        (f["log_volume"] - f["log_volume"].mean()) / f["log_volume"].std(),
    ])
    memoria = _memoria_pelt()
    faltando = [p for p in penalidades if (mtime, short_name, p) not in memoria]
    if faltando:
        with st.spinner(f"Calculando o PELT para {short_name} (só na primeira vez)…"):
            try:
                n_jobs = min(len(faltando), os.cpu_count() or 1)
                resultados = Parallel(n_jobs=n_jobs)(delayed(_pelt)(signal, p) for p in faltando)
            except Exception:  # sem paralelismo disponível: mesmo cálculo, em série
                resultados = [_pelt(signal, p) for p in faltando]
        for p, r in zip(faltando, resultados):
            memoria[(mtime, short_name, p)] = r
    return {p: [f.iloc[b]["date"] for b in memoria[(mtime, short_name, p)]] for p in penalidades}


@st.cache_data(show_spinner="Ajustando o Isolation Forest…")
def ajustar_isolation_forest(contaminacao: float, mtime: float) -> pd.DataFrame:
    """Seção 18: 300 árvores, random_state=42, mesmas features e mesma ordem de linhas."""
    df = preparar_base(mtime)
    model_data = df.dropna(subset=FEATURES)[["market_hash_name", "date", "short_name", *FEATURES]].copy()
    iso = IsolationForest(n_estimators=300, contamination=contaminacao, random_state=42, n_jobs=-1)
    model_data["if_label"] = iso.fit_predict(model_data[FEATURES])
    model_data["if_score"] = -iso.decision_function(model_data[FEATURES])
    model_data["if_candidate"] = model_data["if_label"] == -1
    return model_data


def combinar_if_mad(model_data: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    md = model_data.merge(df[["market_hash_name", "date", "robust_candidate"]], on=["market_hash_name", "date"], how="left")
    md["both"] = md["if_candidate"] & md["robust_candidate"]
    return md


def resumo_if(md: pd.DataFrame) -> pd.DataFrame:
    return md.groupby("short_name").agg(
        observacoes=("date", "size"),
        eventos_if=("if_candidate", "sum"),
        eventos_robustos=("robust_candidate", "sum"),
        sobreposicao=("both", "sum"),
    )


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
        "date", "short_name", "price_median", "volume", "log_return", "robust_score", "if_score",
        "robust_candidate", "if_candidate", "integrated_rank",
    ]]


@st.cache_data(show_spinner=False)
def estudo_evento(mtime: float):
    """Seção 20: janela de ±30 dias do Trade Up com base em D−1 e volume mediana × mediana."""
    df = preparar_base(mtime)
    window = df[(df["date"] >= EVENT_DATE - pd.Timedelta(days=30)) & (df["date"] <= EVENT_DATE + pd.Timedelta(days=30))]
    event_rows, curvas = [], []
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
        curvas.append(g[["date", "short_name", "price_median", "event_index"]])
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
    return pd.DataFrame(event_rows), pd.concat(curvas, ignore_index=True)


def classificar_evento(data: pd.Timestamp, short_name: str) -> tuple:
    """Classificação contextual manual registrada no TCC (tabela de agrupamento do ranking integrado).

    Retorna (grupo, rótulo curto para a mesa de triagem, texto da tabela do TCC).
    """
    if pd.Timestamp("2025-10-22") <= data <= pd.Timestamp("2025-11-02"):
        return ("Choque comum", "Atualização Trade Up (22/10/2025)",
                "Choque comum; associado a contexto legítimo conhecido: atualização oficial de 22/10/2025 (Trade Up)")
    if data == pd.Timestamp("2024-10-03"):
        return ("Choque comum", "Atualização The Armory (02/10/2024)",
                "Choque comum; associado a contexto legítimo conhecido: atualização oficial The Armory, de 02/10/2024 (03/10 em UTC)")
    if short_name.startswith("M4A1-S") and data in (pd.Timestamp("2025-06-24"), pd.Timestamp("2025-06-25")):
        return ("Movimento próprio do item", "Próprio do item: dado ou venda isolada?",
                "Movimento próprio do item; sem explicação observada; hipótese de problema de dado ou venda isolada")
    if short_name.startswith("AK-47") and data == pd.Timestamp("2025-02-01"):
        return ("Choque comum", "Choque comum: causa não confirmada",
                "Choque comum; coincide com a atualização de 29/01/2025 e com relatos de falha no mercado, "
                "sem confirmação oficial")
    return ("Não classificado", "Fora da tabela do TCC",
            "Fora da classificação do TCC; exige investigação contextual")


# =============================================================================
# Formatação (padrão brasileiro)
# =============================================================================
def br(valor, casas: int = 0, sinal: bool = False) -> str:
    if valor is None or pd.isna(valor):
        return "—"
    texto = f"{valor:{'+' if sinal else ''},.{casas}f}"
    return texto.replace(",", "§").replace(".", ",").replace("§", ".")


def br_data(ts) -> str:
    return pd.Timestamp(ts).strftime("%d/%m/%Y")


def estilo_br(tabela: pd.DataFrame, formatos: dict):
    """Styler com vírgula decimal e ponto de milhar; a ordenação continua numérica."""
    return tabela.style.format(formatos, decimal=",", thousands=".", na_rep="—")


def delta_tcc(atual: int, referencia: int):
    return None if atual == referencia else f"{atual - referencia:+d} vs. TCC ({referencia})"


# =============================================================================
# Gráficos (Altair)
# =============================================================================
def grafico_serie(s: pd.DataFrame, quebras: list) -> tuple:
    """Dois painéis com o mesmo eixo de datas (preço em cima, volume embaixo); nunca dois eixos y no mesmo gráfico."""
    s = s.assign(
        x=s["date"] + MEIO_DIA,
        retorno_pct=100 * s["log_return"],
        data_txt=s["date"].dt.strftime("%d/%m/%Y"),
    )
    inicio, fim = s["date"].min(), s["date"].max()
    cand_ret = s[s["return_candidate"]].assign(tipo=LEG_CAND)
    cand_vol = s[s["volume_candidate"]].assign(tipo=LEG_CAND)
    pelt = pd.DataFrame({"date": pd.to_datetime(quebras)})
    pelt = pelt[(pelt["date"] >= inicio) & (pelt["date"] <= fim)]
    pelt = pelt.assign(x=pelt["date"] + MEIO_DIA, tipo=LEG_PELT, data_txt=pelt["date"].dt.strftime("%d/%m/%Y"))
    valve = ATUALIZACOES[(ATUALIZACOES["date"] >= inicio) & (ATUALIZACOES["date"] <= fim)]
    valve = valve.assign(x=valve["date"] + MEIO_DIA, tipo=LEG_VALVE)

    escala_cor = alt.Scale(domain=[LEG_CAND, LEG_PELT, LEG_VALVE], range=[VERMELHO, CINZA, VERDE])
    cor = alt.Color("tipo:N", scale=escala_cor,
                    legend=alt.Legend(title=None, orient="top", direction="horizontal", labelLimit=320))
    cor_sem_legenda = alt.Color("tipo:N", scale=escala_cor, legend=None)
    hover = alt.selection_point(fields=["x"], nearest=True, on="pointerover", empty=False, clear="pointerout")
    hover_v = alt.selection_point(fields=["x"], nearest=True, on="pointerover", empty=False, clear="pointerout")
    eixo_x = alt.X("x:T", title=None, axis=alt.Axis(format="%m/%Y", labelAngle=0))  # mesmos dados nos dois painéis
    largura_eixo_y = 60  # mesma largura de eixo nos dois painéis para as datas ficarem alinhadas

    def regras(cor_regra, com_rotulo):
        camadas = [
            alt.Chart(pelt).mark_rule(strokeDash=[5, 4], strokeWidth=1.5).encode(
                x=eixo_x, color=cor_regra, tooltip=[alt.Tooltip("data_txt:N", title="Quebra PELT (1º dia do novo regime)")]),
            alt.Chart(valve).mark_rule(strokeWidth=2).encode(
                x=eixo_x, color=cor_regra, tooltip=[alt.Tooltip("rotulo:N", title="Atualização oficial")]),
        ]
        if com_rotulo:
            camadas.append(alt.Chart(valve).mark_text(align="left", dx=5, fontSize=11, color=CINZA_TEXTO).encode(
                x=eixo_x, y=alt.value(10), text="rotulo:N"))
        return camadas

    def seletor(chart, param, tooltip):
        return chart.mark_rule(color=CINZA, strokeWidth=1).encode(
            opacity=alt.when(param).then(alt.value(0.45)).otherwise(alt.value(0)), tooltip=tooltip).add_params(param)

    dica_dia = [
        alt.Tooltip("data_txt:N", title="Data"),
        alt.Tooltip("price_median:Q", title="Preço (US$)", format=",.2f"),
        alt.Tooltip("volume:Q", title="Volume (unid.)", format=",d"),
        alt.Tooltip("retorno_pct:Q", title="Retorno log. (%)", format="+.2f"),
        alt.Tooltip("z_return:Q", title="z robusto do retorno", format="+.2f"),
        alt.Tooltip("z_volume:Q", title="z robusto do log-volume", format="+.2f"),
    ]

    base_p = alt.Chart(s).encode(x=eixo_x)
    linha = base_p.mark_line(color=AZUL, strokeWidth=2).encode(
        y=alt.Y("price_median:Q", title="Preço diário (US$)", scale=alt.Scale(zero=False),
                axis=alt.Axis(minExtent=largura_eixo_y, maxExtent=largura_eixo_y)))
    ponto_hover = base_p.mark_point(filled=True, size=60, color=AZUL).encode(
        y="price_median:Q", opacity=alt.when(hover).then(alt.value(1)).otherwise(alt.value(0)))
    cands_p = alt.Chart(cand_ret).mark_point(filled=True, size=80, stroke="white", strokeWidth=1, opacity=1).encode(
        x=eixo_x, y="price_median:Q", color=cor,
        tooltip=[alt.Tooltip("data_txt:N", title="Candidato de retorno"),
                 alt.Tooltip("retorno_pct:Q", title="Retorno log. (%)", format="+.2f"),
                 alt.Tooltip("z_return:Q", title="z robusto", format="+.2f")])
    painel_preco = alt.layer(*regras(cor, True), linha, seletor(base_p, hover, dica_dia), ponto_hover, cands_p
                             ).properties(height=320)

    base_v = alt.Chart(s).encode(x=eixo_x)
    area = base_v.mark_area(color=AZUL_CLARO, opacity=0.85, interpolate="step-after").encode(
        y=alt.Y("volume:Q", title="Volume (unid./dia)", axis=alt.Axis(minExtent=largura_eixo_y, maxExtent=largura_eixo_y)))
    cands_v = alt.Chart(cand_vol).mark_point(filled=True, size=80, stroke="white", strokeWidth=1, opacity=1).encode(
        x=eixo_x, y="volume:Q", color=cor_sem_legenda,
        tooltip=[alt.Tooltip("data_txt:N", title="Candidato de volume"),
                 alt.Tooltip("volume:Q", title="Volume (unid.)", format=",d"),
                 alt.Tooltip("z_volume:Q", title="z robusto", format="+.2f")])
    painel_volume = alt.layer(*regras(cor_sem_legenda, False), area, seletor(base_v, hover_v, dica_dia), cands_v
                              ).properties(height=170)
    return painel_preco, painel_volume


def grafico_evento(curvas: pd.DataFrame) -> alt.LayerChart:
    c = curvas.assign(x=curvas["date"] + MEIO_DIA, data_txt=curvas["date"].dt.strftime("%d/%m/%Y"))
    larga = c.pivot(index="x", columns="short_name", values="event_index")
    larga = larga[[s for s in ORDEM_SKINS if s in larga.columns]]
    apelidos = {nome: f"s{i}" for i, nome in enumerate(larga.columns)}
    larga = larga.rename(columns=apelidos).reset_index()
    larga["data_txt"] = (larga["x"] - MEIO_DIA).dt.strftime("%d/%m/%Y")

    cor = alt.Color("short_name:N", title=None, scale=alt.Scale(domain=ORDEM_SKINS, range=CORES_SKIN),
                    legend=alt.Legend(orient="top", direction="horizontal", columns=3))
    eixo_x = alt.X("x:T", title=None, axis=alt.Axis(format="%d/%m", labelAngle=0))
    hover = alt.selection_point(fields=["x"], nearest=True, on="pointerover", empty=False, clear="pointerout")

    evento = pd.DataFrame({"x": [EVENT_DATE + MEIO_DIA], "rotulo": ["Trade Up (22/10/2025)"]})
    regra_evento = alt.Chart(evento).mark_rule(color=VERDE, strokeWidth=2, strokeDash=[6, 4]).encode(x="x:T")
    rotulo_evento = alt.Chart(evento).mark_text(align="left", dx=5, fontSize=11, color=CINZA_TEXTO).encode(
        x="x:T", y=alt.value(10), text="rotulo:N")
    regra_100 = alt.Chart(pd.DataFrame({"y": [100]})).mark_rule(color=CINZA, strokeDash=[2, 3]).encode(y="y:Q")
    linhas = alt.Chart(c).mark_line(strokeWidth=2).encode(
        x=eixo_x, y=alt.Y("event_index:Q", title="Índice de preço (base 100 = 21/10/2025, D−1)"), color=cor)
    pontos = alt.Chart(c).mark_point(filled=True, size=50).encode(
        x=eixo_x, y="event_index:Q", color=cor, opacity=alt.when(hover).then(alt.value(1)).otherwise(alt.value(0)))
    seletor = alt.Chart(larga).mark_rule(color=CINZA, strokeWidth=1).encode(
        x=eixo_x,
        opacity=alt.when(hover).then(alt.value(0.45)).otherwise(alt.value(0)),
        tooltip=[alt.Tooltip("data_txt:N", title="Data")]
        + [alt.Tooltip(f"{apelido}:Q", title=nome, format=".1f") for nome, apelido in apelidos.items()],
    ).add_params(hover)
    return alt.layer(regra_100, regra_evento, rotulo_evento, linhas, seletor, pontos).properties(height=380)


def grafico_sensibilidade(dados: pd.DataFrame, x: str, x_titulo: str, series: dict, atual: float,
                          rotulo_atual: str, formato_x: str, destaque: str | None = None) -> alt.LayerChart:
    """Uma curva por medida, com o valor atual do slider marcado por uma linha vertical."""
    longo = dados.melt(id_vars=[x] + ([destaque] if destaque else []), value_vars=list(series),
                       var_name="medida", value_name="valor")
    longo["medida"] = longo["medida"].map(series)
    nomes = list(series.values())
    cor = alt.Color("medida:N", scale=alt.Scale(domain=nomes, range=CORES_SKIN[: len(nomes)]),
                    legend=alt.Legend(title=None, orient="top") if len(nomes) > 1 else None)
    eixo_x = alt.X(f"{x}:Q", title=x_titulo, axis=alt.Axis(format=formato_x))
    tooltip = [alt.Tooltip(f"{x}:Q", title=x_titulo, format=formato_x), alt.Tooltip("medida:N", title="Medida"),
               alt.Tooltip("valor:Q", title="Valor", format=",d")]
    linhas = alt.Chart(longo).mark_line(strokeWidth=2).encode(x=eixo_x, y=alt.Y("valor:Q", title="Eventos"), color=cor)
    if destaque:
        pontos = alt.Chart(longo).mark_point(filled=True).encode(
            x=eixo_x, y="valor:Q", color=cor, tooltip=tooltip,
            size=alt.when(alt.datum[destaque]).then(alt.value(90)).otherwise(alt.value(25)))
    else:
        pontos = alt.Chart(longo).mark_point(filled=True, size=70).encode(x=eixo_x, y="valor:Q", color=cor, tooltip=tooltip)
    camadas = [linhas, pontos]
    if len(nomes) > 1:  # rótulos diretos no último ponto (≤ 4 séries)
        ultimo = longo[longo[x] == longo[x].max()]
        camadas.append(alt.Chart(ultimo).mark_text(align="left", dx=6, fontSize=11, color=CINZA_TEXTO).encode(
            x=eixo_x, y="valor:Q", text=alt.Text("valor:Q", format=",d")))
    marca = pd.DataFrame({x: [atual], "rotulo": [rotulo_atual]})
    camadas.append(alt.Chart(marca).mark_rule(color=VERMELHO, strokeWidth=1.5, strokeDash=[4, 3]).encode(x=f"{x}:Q"))
    camadas.append(alt.Chart(marca).mark_text(align="left", dx=5, fontSize=11, color=CINZA_TEXTO).encode(
        x=f"{x}:Q", y=alt.value(10), text="rotulo:N"))
    return alt.layer(*camadas).properties(height=260)


# =============================================================================
# Interface
# =============================================================================
def restaurar_padrao():
    for chave, valor in PADRAO.items():
        st.session_state[chave] = valor


def barra_lateral():
    for chave, valor in PADRAO.items():
        st.session_state.setdefault(chave, valor)
    st.session_state.setdefault("skin", ORDEM_SKINS[0])
    with st.sidebar:
        st.header("Parâmetros")
        skin = st.selectbox("Skin", ORDEM_SKINS, key="skin")
        limiar = st.slider(
            "Limiar do baseline robusto (MAD)", 3.0, 6.0, step=0.1, format="%.1f", key="limiar",
            help="Candidato quando |z| do retorno ou do log-volume ≥ limiar, com z = (x − mediana) / (1,4826 × MAD).")
        penalidade = st.slider(
            "Penalização do PELT", 5, 20, step=1, key="penalidade",
            help="Custo de abrir um novo regime: quanto maior, menos quebras. Sinal = log-preço e log-volume padronizados; "
                 "custo RBF; segmento mínimo de 14 dias.")
        contaminacao = st.slider(
            "Contaminação do Isolation Forest", 0.01, 0.05, step=0.005, format="%.3f", key="contaminacao",
            help="Fração das observações marcada como anômala pelo Isolation Forest (300 árvores, semente 42).")
        st.button("Restaurar parâmetros do TCC", on_click=restaurar_padrao, icon=":material/restart_alt:", width="stretch")

        limiar, contaminacao = round(float(limiar), 1), round(float(contaminacao), 3)
        if (limiar, penalidade, contaminacao) == (PADRAO["limiar"], PADRAO["penalidade"], PADRAO["contaminacao"]):
            st.success("Parâmetros do TCC: os números coincidem com as tabelas do texto.", icon=":material/check_circle:")
        else:
            st.warning("Parâmetros alterados: exploração de sensibilidade. Os números diferem das tabelas do TCC.",
                       icon=":material/tune:")

        with st.expander("Como os métodos funcionam"):
            st.markdown(
                "**Baseline robusto (MAD).** Para cada skin, o escore robusto é\n\n"
                "$z = \\dfrac{x - \\text{mediana}}{1{,}4826 \\times \\text{MAD}}$\n\n"
                "aplicado ao retorno logarítmico e ao log-volume. O fator 1,4826 torna o MAD comparável ao desvio-padrão "
                "sob normalidade.\n\n"
                "**PELT.** Busca o conjunto de quebras que minimiza custo + penalização × nº de quebras.\n\n"
                "**Isolation Forest.** Isola cada observação com cortes aleatórios; pontos isolados com poucos cortes "
                "recebem escore alto. Features: z do retorno, do log-volume, da variação do log-volume e da volatilidade "
                "móvel de 30 dias.\n\n"
                "**Ranking integrado.** Média dos postos percentuais do escore robusto e do escore do Isolation Forest."
            )
    return skin, limiar, penalidade, contaminacao


def aba_visao_geral(df, md, df_ref, md_ref, skin, penalidade, quebras):
    st.subheader("Cesta analisada")
    n_mad, n_if, n_both = int(df["robust_candidate"].sum()), int(md["if_candidate"].sum()), int(md["both"].sum())
    ref_mad, ref_if, ref_both = int(df_ref["robust_candidate"].sum()), int(md_ref["if_candidate"].sum()), int(md_ref["both"].sum())
    c = st.columns(6)
    c[0].metric("Observações", br(len(df)), help="Pares skin × dia")
    c[1].metric("Skins", br(df["short_name"].nunique()), help="Todas no desgaste Field-Tested")
    c[2].metric("Período", f"{df['date'].min():%Y}–{df['date'].max():%y}",
                help=f"{br_data(df['date'].min())} a {br_data(df['date'].max())}")
    c[3].metric("Candidatos MAD", br(n_mad), delta=delta_tcc(n_mad, ref_mad), delta_color="off",
                help=f"Baseline robusto: {br(100 * n_mad / len(df), 2)}% das observações")
    c[4].metric("Eventos IF", br(n_if), delta=delta_tcc(n_if, ref_if), delta_color="off",
                help=f"Isolation Forest. Avaliadas {br(len(md))} observações: os 15 primeiros dias de cada série "
                     "não têm volatilidade móvel.")
    c[5].metric("Convergentes", br(n_both), delta=delta_tcc(n_both, ref_both), delta_color="off",
                help=f"Marcados pelos dois métodos: {br(100 * n_both / n_if, 0) if n_if else '—'}% dos eventos do IF "
                     "também são candidatos MAD")

    st.markdown("**Resumo por skin** — preços em US$; contagens com os parâmetros atuais")
    cont, rif = contagem_mad(df), resumo_if(md)
    precos = df.groupby("short_name").agg(
        obs=("date", "size"), mediana=("price_median", "median"), minimo=("price_median", "min"),
        maximo=("price_median", "max"), vol=("volume", "median"))
    resumo = pd.DataFrame({
        "Obs.": precos["obs"], "Preço mediano": precos["mediana"], "Mín.": precos["minimo"], "Máx.": precos["maximo"],
        "Vol. med.": precos["vol"], "Cand. retorno": cont["retorno_candidato"], "Cand. volume": cont["volume_candidato"],
        "União MAD": cont["uniao"], "% obs.": cont["percentual_uniao"], "Eventos IF": rif["eventos_if"],
        "Ambos": rif["sobreposicao"],
    }).reindex(ORDEM_SKINS)
    total = resumo.sum()
    total[["Preço mediano", "Mín.", "Máx.", "Vol. med."]] = np.nan
    total["% obs."] = 100 * cont["uniao"].sum() / cont["observacoes"].sum()
    resumo.loc["Total"] = total
    inteiros = ["Obs.", "Vol. med.", "Cand. retorno", "Cand. volume", "União MAD", "Eventos IF", "Ambos"]
    for coluna in resumo.columns:
        casas = 0 if coluna in inteiros else 2
        resumo[coluna] = [br(v, casas) if pd.notna(v) else "" for v in resumo[coluna]]
    resumo = resumo.rename_axis("Skin").reset_index()
    st.dataframe(resumo, hide_index=True, column_config={
        c: st.column_config.TextColumn(alignment="right") for c in resumo.columns if c != "Skin"})
    st.caption("Vol. med.: mediana de unidades vendidas por dia. União MAD: dias com candidato de retorno ou de volume, "
               "contados uma vez. Ambos: candidatos MAD também marcados pelo Isolation Forest.")

    st.divider()
    st.subheader(f"Série diária — {skin}")
    serie = df[df["short_name"] == skin]
    d_min, d_max = serie["date"].min().date(), serie["date"].max().date()
    janela = st.slider("Janela de datas exibida", min_value=d_min, max_value=d_max, value=(d_min, d_max),
                       format="DD/MM/YYYY", key=f"janela_{skin}")
    vis = serie[(serie["date"] >= pd.Timestamp(janela[0])) & (serie["date"] <= pd.Timestamp(janela[1]))]
    sk_md = md[md["short_name"] == skin]
    c = st.columns(5)
    c[0].metric("Observações", br(len(serie)))
    c[1].metric("Candidatos de retorno", br(int(serie["return_candidate"].sum())))
    c[2].metric("Candidatos de volume", br(int(serie["volume_candidate"].sum())))
    c[3].metric("Eventos do IF", br(int(sk_md["if_candidate"].sum())))
    c[4].metric(f"Quebras PELT (pen. {penalidade})", br(len(quebras)))
    painel_preco, painel_volume = grafico_serie(vis, quebras)
    st.altair_chart(painel_preco, width="stretch")
    st.altair_chart(painel_volume, width="stretch")
    st.caption(
        "Pontos vermelhos: candidatos do baseline robusto — de retorno no painel de preço e de volume no painel de volume. "
        "Linhas tracejadas cinza: quebras de regime do PELT (primeiro dia do novo segmento). Linhas verdes: atualizações "
        "oficiais da Valve. Passe o mouse sobre o gráfico para ver os valores do dia. Marcações são descritivas e não "
        "confirmam manipulação."
    )
    st.markdown(f"**Quebras de regime do PELT (penalização {penalidade}):** "
                + (" · ".join(br_data(d) for d in quebras) if quebras else "nenhuma"))
    if skin != ORDEM_SKINS[0]:
        st.caption("No TCC, o PELT é reportado para a AK-47 Asiimov. Para esta skin, a mesma configuração é aplicada "
                   "de forma exploratória.")

    with st.expander(f"Ver os {int(serie['robust_candidate'].sum())} candidatos MAD de {skin}"):
        cand = serie[serie["robust_candidate"]].merge(
            sk_md[["date", "if_candidate"]], on="date", how="left")
        tabela = pd.DataFrame({
            "Data": cand["date"].dt.strftime("%d/%m/%Y"),
            "Tipo": np.select([cand["return_candidate"] & cand["volume_candidate"], cand["return_candidate"]],
                              ["Retorno e volume", "Retorno"], "Volume"),
            "Preço (US$)": cand["price_median"], "Volume": cand["volume"],
            "Retorno log. (%)": 100 * cand["log_return"], "z retorno": cand["z_return"], "z volume": cand["z_volume"],
            "Também no IF": cand["if_candidate"].map({True: "Sim", False: "Não"}).fillna("Fora do IF"),
        })
        st.dataframe(estilo_br(tabela, {"Preço (US$)": "{:,.2f}", "Volume": "{:,.0f}", "Retorno log. (%)": "{:+,.2f}",
                                        "z retorno": "{:+,.2f}", "z volume": "{:+,.2f}"}), hide_index=True)


def aba_triagem(top: pd.DataFrame, limiar: float, contaminacao: float):
    st.subheader("Mesa de triagem — 25 observações de maior prioridade")
    st.markdown(
        "Cada observação (skin × dia) recebe dois postos percentuais: o do **escore robusto** "
        "(maior |z| entre retorno e log-volume) e o do **escore do Isolation Forest**. "
        "O **posto integrado** é a média dos dois; quanto mais perto de 1, maior a prioridade de inspeção. "
        "A classificação contextual é a registrada no TCC após consulta às notas oficiais da Valve."
    )
    classes = top.apply(lambda r: classificar_evento(r["date"], r["short_name"]), axis=1, result_type="expand")
    tabela = pd.DataFrame({
        "Posição": np.arange(1, len(top) + 1),
        "Data": top["date"].dt.strftime("%d/%m/%Y").to_numpy(),
        "Skin": top["short_name"].to_numpy(),
        "Contexto": classes[1].to_numpy(),
        "Posto integrado": top["integrated_rank"].to_numpy(),
        "Escore robusto": top["robust_score"].to_numpy(),
        "Escore IF": top["if_score"].to_numpy(),
        "MAD": np.where(top["robust_candidate"].astype(bool), "Sim", "Não"),
        "IF": np.where(top["if_candidate"].astype(bool), "Sim", "Não"),
        "Retorno (%)": 100 * top["log_return"].to_numpy(),
        "Preço (US$)": top["price_median"].to_numpy(),
        "Volume": top["volume"].to_numpy(),
        "Grupo": classes[0].to_numpy(),
        "Classificação no TCC": classes[2].to_numpy(),
    })
    n_oficial = int((tabela["Grupo"] == "Choque comum").sum())
    n_sem = int((tabela["Grupo"] == "Movimento próprio do item").sum())
    n_conv = int(((tabela["MAD"] == "Sim") & (tabela["IF"] == "Sim")).sum())
    c = st.columns(4)
    c[0].metric("Observações priorizadas", br(len(tabela)))
    c[1].metric("Em dias de choque comum", br(n_oficial),
                help="Trade Up (22/10/2025), The Armory (02/10/2024) e o choque de 01/02/2025. "
                     "A amplitude é medida no índice de mercado de 1.412 skins (aba Mercado & Regra do app principal).")
    c[2].metric("Próprias do item", br(n_sem), help="Movimento só do item; permanecem para conferência com outra fonte")
    c[3].metric("Marcadas pelos dois métodos", br(n_conv),
                help=f"Com limiar {br(limiar, 1)} e contaminação {br(contaminacao, 3)}")

    filtro = st.radio("Mostrar", ["Todas", "Em dias de choque comum", "Próprias do item"],
                      horizontal=True, label_visibility="collapsed")
    if filtro == "Em dias de choque comum":
        vis = tabela[tabela["Grupo"] == "Choque comum"]
    elif filtro == "Próprias do item":
        vis = tabela[tabela["Grupo"] != "Choque comum"]
    else:
        vis = tabela

    def cor_linha(linha):
        if linha["Grupo"] == "Choque comum":
            return [""] * len(linha)
        return ["background-color: rgba(227, 73, 72, 0.13)"] * len(linha)

    estilo = estilo_br(vis, {"Preço (US$)": "{:,.2f}", "Volume": "{:,.0f}", "Retorno (%)": "{:+,.2f}",
                             "Escore robusto": "{:,.2f}", "Escore IF": "{:,.3f}", "Posto integrado": "{:,.4f}"}
                       ).apply(cor_linha, axis=1)
    visiveis = [c for c in tabela.columns if c not in ("Grupo", "Classificação no TCC")]
    st.dataframe(estilo, hide_index=True, height=35 * (len(vis) + 1) + 3, column_order=visiveis,
                 column_config={"Posição": st.column_config.NumberColumn("Nº", width="small"),
                                "Contexto": st.column_config.TextColumn(width=255),
                                "Retorno (%)": st.column_config.Column(help="Retorno logarítmico diário × 100"),
                                "MAD": st.column_config.TextColumn(width="small"),
                                "IF": st.column_config.TextColumn(width="small")})
    st.caption("Linhas destacadas: movimento próprio do item (ou fora da tabela do TCC). O índice de mercado, a regra de "
               "possível sinal de manipulação e o piso de detecção estão na aba Mercado & Regra do app principal. "
               "Clique no cabeçalho de uma coluna para ordenar.")

    st.markdown("**Agrupamento por período** (como na tabela do TCC)")
    grupos = []
    for texto, g in tabela.groupby("Classificação no TCC", sort=False):
        datas = pd.to_datetime(g["Data"], format="%d/%m/%Y")
        periodo = br_data(datas.min()) if datas.min() == datas.max() else f"{br_data(datas.min())} a {br_data(datas.max())}"
        armas = g["Skin"].str.split(" | ", regex=False).str[0].value_counts()
        armas = sorted(armas.items(), key=lambda kv: (-kv[1], kv[0]))
        com_contagem = len(armas) > 1 and any(n > 1 for _, n in armas)
        itens = ", ".join(f"{a} ({n})" if com_contagem else a for a, n in armas)
        grupos.append({"Período": periodo, "Obs.": len(g), "Itens": itens, "Classificação": texto})
    st.table(pd.DataFrame(grupos), hide_index=True)

    st.download_button("Baixar a tabela (CSV)", tabela.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                       file_name="mesa_de_triagem_top25.csv", mime="text/csv", icon=":material/download:")
    with st.expander("Por que o ranking não muda quando mexo no limiar ou na contaminação?"):
        st.markdown(
            "O posto integrado usa os **escores contínuos**, não as marcações binárias. O escore robusto não depende do "
            "limiar, e a contaminação do Isolation Forest só desloca o ponto de corte do escore (as árvores são as mesmas, "
            "com semente fixa). Postos percentuais não mudam com esse deslocamento. Por isso, a lista das 25 observações é "
            "estável; o que muda são as colunas **MAD** e **IF**, que dizem se cada observação ultrapassa o corte escolhido."
        )


def aba_evento_sensibilidade(df, base, skin, limiar, penalidade, contaminacao, pelt_grade, sens_if, mtime):
    st.subheader("Estudo de evento — atualização Trade Up de 22/10/2025")
    resumo, curvas = estudo_evento(mtime)
    st.altair_chart(grafico_evento(curvas), width="stretch")
    st.caption("Base 100 no último preço antes da atualização (21/10/2025, D−1). Linha tracejada verde: data da atualização. "
               "Passe o mouse para ver o índice das cinco skins no dia.")
    tabela = resumo.set_index("item").reindex(ORDEM_SKINS).rename_axis("Skin").reset_index()
    tabela = pd.DataFrame({
        "Skin": tabela["Skin"], "Preço D−1 (US$)": tabela["preco_base"],
        "D0 (%)": tabela["retorno_d0_%"], "D+7 (%)": tabela["retorno_7d_%"], "D+30 (%)": tabela["retorno_30d_%"],
        "Vol. mediano antes": tabela["volume_mediano_antes"], "Vol. mediano depois": tabela["volume_mediano_depois"],
        "Razão": tabela["razao_volume_mediano"],
        "Vol. máx. antes": tabela["volume_maximo_antes"], "Vol. máx. depois": tabela["volume_maximo_depois"],
    })
    st.dataframe(estilo_br(tabela, {"Preço D−1 (US$)": "{:,.2f}", "D0 (%)": "{:+,.1f}", "D+7 (%)": "{:+,.1f}",
                                    "D+30 (%)": "{:+,.1f}", "Vol. mediano antes": "{:,.1f}",
                                    "Vol. mediano depois": "{:,.1f}", "Razão": "{:,.1f}",
                                    "Vol. máx. antes": "{:,.0f}", "Vol. máx. depois": "{:,.0f}"}), hide_index=True)
    altas7 = resumo[resumo["retorno_7d_%"] > 0]["retorno_7d_%"]
    st.markdown(
        f"- No próprio dia 22/10, os preços variaram de **{br(resumo['retorno_d0_%'].min(), 1)}%** a "
        f"**{br(resumo['retorno_d0_%'].max(), 1)}%** em relação a D−1.\n"
        f"- Até D+7, {len(altas7)} skins acumularam alta de **{br(altas7.min(), 1)}%** a **{br(altas7.max(), 1)}%**; "
        f"até D+30, a variação foi de **{br(resumo['retorno_30d_%'].min(), 1)}%** a "
        f"**{br(resumo['retorno_30d_%'].max(), 1)}%**.\n"
        f"- O volume mediano (D−30 a D−1 contra D0 a D+30) aumentou de **{br(resumo['razao_volume_mediano'].min(), 1)}** a "
        f"**{br(resumo['razao_volume_mediano'].max(), 1)} vezes**.\n"
        "- A simultaneidade em skins diferentes favorece a hipótese de **choque comum do ecossistema**; a janela é "
        "descritiva e não isola o efeito da atualização."
    )

    st.divider()
    st.subheader("Sensibilidade dos parâmetros")
    st.caption("Marcadores maiores e tabelas: valores avaliados no TCC. Linha tracejada vermelha: valor atual do slider.")
    col_mad, col_pelt = st.columns(2)
    with col_mad:
        st.markdown("**Baseline robusto: eventos por limiar** (5 skins)")
        fina = sensibilidade_mad(base, np.round(np.arange(3.0, 6.0001, 0.1), 1))
        fina["no_tcc"] = fina["limiar"].isin(GRADE_MAD)
        st.altair_chart(grafico_sensibilidade(fina, "limiar", "Limiar |z|", {"eventos": "Eventos MAD"}, limiar,
                                              f"atual: {br(limiar, 1)}", ".1f", destaque="no_tcc"), width="stretch")
        grade = sensibilidade_mad(base, sorted(set(GRADE_MAD) | {limiar}))
        grade["Situação"] = np.where(grade["limiar"] == limiar, "atual", "")
        st.dataframe(estilo_br(grade.rename(columns={"limiar": "Limiar", "eventos": "Eventos", "percentual": "% das obs."}),
                               {"Limiar": "{:,.1f}", "Eventos": "{:,.0f}", "% das obs.": "{:,.2f}"}), hide_index=True)
    with col_pelt:
        st.markdown(f"**PELT: quebras de regime por penalização** ({skin})")
        pens = sorted(pelt_grade)
        dados = pd.DataFrame({"penalizacao": pens, "pontos": [len(pelt_grade[p]) for p in pens]})
        dados["no_tcc"] = dados["penalizacao"].isin(GRADE_PELT)
        st.altair_chart(grafico_sensibilidade(dados, "penalizacao", "Penalização", {"pontos": "Quebras de regime"},
                                              penalidade, f"atual: {penalidade}", "d", destaque="no_tcc"),
                        width="stretch")
        em_todas = set.intersection(*(set(pelt_grade[p]) for p in GRADE_PELT))
        linhas = ["| Pen. | Pontos | Datas (1º dia do novo segmento) |", "|---:|---:|---|"]
        for p in pens:
            datas = ", ".join(f"**{d:%d/%m/%y}**" if d in em_todas else f"{d:%d/%m/%y}" for d in pelt_grade[p])
            linhas.append(f"| {p}{' (atual)' if p == penalidade else ''} | {len(pelt_grade[p])} | {datas} |")
        st.markdown("\n".join(linhas))
        st.caption("Em negrito: datas presentes em todas as penalizações avaliadas no TCC.")

    st.markdown("**Isolation Forest: eventos por contaminação** (5 skins)")
    col_g, col_t = st.columns([3, 2])
    with col_g:
        st.altair_chart(grafico_sensibilidade(sens_if, "contaminacao", "Contaminação",
                                              {"eventos_if": "Eventos do IF", "ambos": "Também candidatos MAD"},
                                              contaminacao, f"atual: {br(contaminacao, 3)}", ".3f"), width="stretch")
    with col_t:
        tab_if = sens_if.rename(columns={"contaminacao": "Contaminação", "eventos_if": "Eventos IF", "ambos": "Também MAD"})
        tab_if["% também MAD"] = 100 * tab_if["Também MAD"] / tab_if["Eventos IF"]
        st.dataframe(estilo_br(tab_if, {"Contaminação": "{:,.3f}", "Eventos IF": "{:,.0f}", "Também MAD": "{:,.0f}",
                                        "% também MAD": "{:,.0f}"}), hide_index=True)
        st.caption(f"Sobreposição calculada com o limiar MAD atual ({br(limiar, 1)}).")


def main():
    st.set_page_config(page_title="Mesa de triagem — CS2", page_icon=":material/query_stats:", layout="wide")
    if not DATA.exists():
        st.error(f"Arquivo não encontrado: {DATA}")
        st.stop()
    mtime = DATA.stat().st_mtime
    skin, limiar, penalidade, contaminacao = barra_lateral()

    st.title("Mesa de triagem de eventos anômalos")
    st.caption("Skins de CS2 no Steam Community Market, 2024–2026 · TCC de Matheus Lima Ribeiro "
               "(Ciência de Dados e IA — IESB) · Orientador: Prof. Sérgio da Costa Côrtes")
    st.info("**Triagem, não prova.** As séries são agregados diários (skin × dia), sem livro de ofertas nem "
            "identificação de contas. Um evento anômalo indica prioridade de investigação contextual, não manipulação "
            "intencional.", icon=":material/info:")

    base = preparar_base(mtime)
    df = marcar_mad(base, limiar)
    md = combinar_if_mad(ajustar_isolation_forest(contaminacao, mtime), df)
    df_ref = marcar_mad(base, PADRAO["limiar"])
    md_ref = combinar_if_mad(ajustar_isolation_forest(PADRAO["contaminacao"], mtime), df_ref)
    top = ranking_integrado(df, md)
    pelt_grade = obter_pelt(base, skin, sorted(set(GRADE_PELT) | {penalidade}), mtime)
    sens_if = pd.DataFrame([
        {"contaminacao": c, "eventos_if": int(m["if_candidate"].sum()), "ambos": int(m["both"].sum())}
        for c in sorted(set(GRADE_IF) | {contaminacao})
        for m in [combinar_if_mad(ajustar_isolation_forest(c, mtime), df)]
    ])

    aba1, aba2, aba3 = st.tabs(["Visão geral e série por skin", "Mesa de triagem — ranking integrado",
                                "Estudo de evento e sensibilidade"])
    with aba1:
        aba_visao_geral(df, md, df_ref, md_ref, skin, penalidade, pelt_grade[penalidade])
    with aba2:
        aba_triagem(top, limiar, contaminacao)
    with aba3:
        aba_evento_sensibilidade(df, base, skin, limiar, penalidade, contaminacao, pelt_grade, sens_if, mtime)

    st.divider()
    st.caption("Fonte: elaboração própria com base em Pruden (2026), dataset público do Kaggle (Apache 2.0), cesta de "
               "5 skins Field-Tested de 01/01/2024 a 15/06/2026. Cálculos idênticos a prototipo/gerar_tabelas_cesta.py.")


if __name__ == "__main__":
    main()
