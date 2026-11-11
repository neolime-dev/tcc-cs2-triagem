/* CS2 Market Triage — frontend (Plotly.js). Todos os números vêm de /api/state (server.py). */
"use strict";

const C = {
  bg: "#0b0e14", bg2: "#131824", line: "#1f293d", line2: "#2a3650", tx: "#e6ebf5", tx2: "#9aa6bd", tx3: "#66728a",
  accent: "#4ea1ff", red: "#ff5a5a", amber: "#ffb547", cyan: "#22d3ee", yellow: "#facc15", green: "#34d399",
  price: "#c9d6ff", vol: "#2c4670",
};
const SKIN_COR = {
  "AK-47 | Asiimov": "#4ea1ff", "AWP | Hyper Beast": "#fb923c", "M4A1-S | Printstream": "#2dd4bf",
  "Desert Eagle | Printstream": "#a3e635", "Glock-18 | Water Elemental": "#e879f9",
};
const MONO = "JetBrains Mono, ui-monospace, monospace";
const UI = "Inter, system-ui, sans-serif";
const PADRAO = { limiar: 4.0, penalidade: 10, contaminacao: 0.02, amplitude: 10, volz: 4.0, reversao: 7 };

const S = { skin: "AK-47 | Asiimov", ...PADRAO, sel: null, filtro: "all", filtroEp: "all", data: null, range: null };
const cache = new Map();
let seq = 0;

/* ---------------- formatação pt-BR ---------------- */
const nf = (d) => new Intl.NumberFormat("pt-BR", { minimumFractionDigits: d, maximumFractionDigits: d });
const NF = [0, 1, 2, 3, 4].map(nf);
const f = (v, d = 0) => (v === null || v === undefined || Number.isNaN(v) ? "—" : NF[d].format(v));
const fs = (v, d = 1) => (v === null || v === undefined ? "—" : (v > 0 ? "+" : v < 0 ? "−" : "") + NF[d].format(Math.abs(v)));
const usd = (v) => (v === null || v === undefined ? "—" : "US$ " + NF[2].format(v));
const dbr = (iso) => { const [y, m, d] = iso.split("-"); return `${d}/${m}/${y}`; };
const dshort = (iso) => { const [y, m, d] = iso.split("-"); return `${d}/${m}/${y.slice(2)}`; };
const dias = (a, b) => Math.round((Date.parse(a) - Date.parse(b)) / 864e5);
const noon = (iso) => iso + " 12:00";
const addDays = (iso, n) => new Date(Date.parse(iso) + n * 864e5).toISOString().slice(0, 10);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const [arma, padrao] = [(s) => s.split(" | ")[0], (s) => s.split(" | ")[1] || ""];
const $ = (id) => document.getElementById(id);
const keyOf = (e) => e && `${e.skin}@${e.date}`;
const isOficial = () => S.limiar === PADRAO.limiar && S.penalidade === PADRAO.penalidade && Math.abs(S.contaminacao - PADRAO.contaminacao) < 1e-9
  && S.amplitude === PADRAO.amplitude && S.volz === PADRAO.volz && S.reversao === PADRAO.reversao;

/* ---------------- API ---------------- */
async function carregar() {
  const qs = new URLSearchParams({ skin: S.skin, limiar: S.limiar.toFixed(1), penalidade: String(S.penalidade), contaminacao: S.contaminacao.toFixed(3),
    amplitude: String(S.amplitude), volz: S.volz.toFixed(1), reversao: String(S.reversao) });
  const k = qs.toString();
  const my = ++seq;
  let data = cache.get(k);
  if (!data) {
    const t = setTimeout(() => $("loader").classList.add("on"), 120);
    try {
      const r = await fetch("/api/state?" + k);
      if (r.status === 503) {  // servidor ainda calculando (ex.: acordando na nuvem): tenta de novo
        setStatus("", "carregando dados…");
        setTimeout(() => { if (my === seq) carregar(); }, 2000);
        return;
      }
      if (!r.ok) throw new Error("HTTP " + r.status);
      data = await r.json();
      cache.set(k, data);
      setStatus("ok", "API online");
    } catch (e) {
      setStatus("err", "falha na API");
      console.error(e);
      return;
    } finally { clearTimeout(t); $("loader").classList.remove("on"); }
  }
  if (my !== seq) return;
  S.data = data;
  render();
}
function setStatus(cls, txt) { $("status").className = "status " + cls; $("status-txt").textContent = txt; }

/* ---------------- render geral ---------------- */
function render() {
  const D = S.data;
  if (!S.sel || !D.top.some((e) => keyOf(e) === S.sel)) {
    const first = D.top.find((e) => e.skin === S.skin);
    S.sel = first ? keyOf(first) : null;
  }
  $("meta-periodo").textContent = `${dbr(D.kpis.inicio)} → ${dbr(D.kpis.fim)}`;
  $("meta-obs").textContent = `${f(D.kpis.observacoes)} obs · ${D.kpis.skins} skins · Field-Tested`;
  renderCards(D); renderMode(); renderKpis(D); renderChart(D); renderTable(D); renderGroups(D); renderDossie(D);
  renderEvento(D); renderSens(D); renderMercado(D);
}

function renderCards(D) {
  const el = $("cards");
  el.innerHTML = D.cards.map((c) => {
    const pts = c.spark, mn = Math.min(...pts), mx = Math.max(...pts);
    const path = pts.map((v, i) => `${(i / (pts.length - 1)) * 84},${28 - ((v - mn) / (mx - mn || 1)) * 26}`).join(" ");
    return `<button class="card ${c.skin === S.skin ? "on" : ""}" style="--rc:${c.cor}" data-skin="${esc(c.skin)}" type="button" aria-pressed="${c.skin === S.skin}">
      <div class="card-top"><span class="rar">${c.raridade}</span><span class="card-alert" title="Candidatos MAD (limiar atual) · eventos IF"><b>${c.alertas_mad}</b> MAD · ${c.alertas_if} IF</span></div>
      <div class="card-name">${esc(arma(c.skin))} <span>| ${esc(padrao(c.skin))}</span></div>
      <div class="card-row">
        <div class="card-stats"><span>Preço mediano</span><span>Vol. mediano</span><span class="v">${usd(c.preco_mediano)}</span><span class="v">${f(c.volume_mediano)}/dia</span></div>
        <svg class="spark" viewBox="0 0 84 30" preserveAspectRatio="none" aria-hidden="true"><polyline points="${path}" fill="none" stroke="${c.cor}" stroke-width="1.5" stroke-linejoin="round" opacity=".9"/></svg>
      </div></button>`;
  }).join("");
}

function renderMode() {
  const b = $("badge-mode"), of = isOficial();
  b.textContent = of ? "PARÂMETROS OFICIAIS DO TCC" : "MODO EXPLORATÓRIO";
  b.classList.toggle("expl", !of);
  b.title = of ? "Limiar 4,0 · penalização 10 · contaminação 0,020 · amplitude 10% · limiar de volume 4,0 · reversão em 7 dias — os números coincidem com a monografia" : "Parâmetros diferentes dos da monografia: resultados exploratórios";
  $("btn-reset").disabled = of;
  $("out-limiar").classList.toggle("changed", S.limiar !== PADRAO.limiar);
  $("out-pen").classList.toggle("changed", S.penalidade !== PADRAO.penalidade);
  $("out-cont").classList.toggle("changed", Math.abs(S.contaminacao - PADRAO.contaminacao) > 1e-9);
  $("out-amp").classList.toggle("changed", S.amplitude !== PADRAO.amplitude);
  $("out-volz").classList.toggle("changed", S.volz !== PADRAO.volz);
  $("out-rev").classList.toggle("changed", S.reversao !== PADRAO.reversao);
}

function renderKpis(D) {
  const k = D.kpis;
  const delta = (v, r) => (v === r ? `<div class="kpi-d">= TCC</div>` : `<div class="kpi-d up">${v > r ? "+" : "−"}${Math.abs(v - r)} vs. TCC (${r})</div>`);
  const sk = D.cards.find((c) => c.skin === S.skin);
  $("kpis").innerHTML = [
    `<div class="kpi"><div class="kpi-l">Observações</div><div class="kpi-v">${f(k.observacoes)}</div><div class="kpi-d">${k.skins} skins × dia · IF: ${f(k.avaliadas_if)}</div></div>`,
    `<div class="kpi"><div class="kpi-l">Candidatos MAD</div><div class="kpi-v" style="color:${C.red}">${f(k.mad)}</div>${delta(k.mad, k.mad_ref)}</div>`,
    `<div class="kpi"><div class="kpi-l">Eventos Isolation Forest</div><div class="kpi-v" style="color:${C.amber}">${f(k.if)}</div>${delta(k.if, k.if_ref)}</div>`,
    `<div class="kpi"><div class="kpi-l">Convergentes (MAD ∩ IF)</div><div class="kpi-v">${f(k.ambos)}</div>${delta(k.ambos, k.ambos_ref)}</div>`,
    `<div class="kpi"><div class="kpi-l">Concordância IF → MAD</div><div class="kpi-v">${k.if ? f((100 * k.ambos) / k.if) : "—"}%</div><div class="kpi-d">dos eventos do IF também são MAD</div></div>`,
    `<div class="kpi"><div class="kpi-l">Quebras PELT · ${esc(arma(S.skin))}</div><div class="kpi-v" style="color:${C.cyan}">${D.pelt.length}</div><div class="kpi-d">penalização ${S.penalidade} · ${sk.alertas_mad} MAD · ${sk.alertas_if} IF</div></div>`,
  ].join("");
}

/* ---------------- gráfico principal (preço + volume sincronizados) ---------------- */
const AX = { gridcolor: "rgba(31,41,61,.75)", zerolinecolor: C.line, linecolor: C.line, tickfont: { family: MONO, size: 10.5, color: C.tx3 } };
const baseLayout = () => ({
  paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)", font: { family: UI, color: C.tx2, size: 11.5 },
  hoverlabel: { bgcolor: "#0f1420", bordercolor: C.line2, font: { family: MONO, size: 11.5, color: C.tx } },
  showlegend: false,
});

function renderChart(D) {
  const s = D.serie, sel = D.top.find((e) => keyOf(e) === S.sel && e.skin === S.skin);
  const card = D.cards.find((c) => c.skin === S.skin);
  $("chart-title").innerHTML = `<span style="color:${card.cor}">●</span> ${esc(S.skin)} <span class="mono" style="color:${C.tx3};font-weight:500;font-size:12.5px">· ${card.raridade} · Field-Tested</span>`;
  const X = s.date.map(noon);
  const idx = (m) => m.map((v, i) => (v ? i : -1)).filter((i) => i >= 0);
  const iRet = idx(s.ret_cand), iIf = idx(s.if_cand);
  const custom = s.date.map((_, i) => [s.z_return[i], s.z_volume[i], s.if_score[i], s.log_return[i] === null ? null : 100 * s.log_return[i]]);
  const pick = (arr, ii) => ii.map((i) => arr[i]);
  const hovP = "<b>%{x|%d/%m/%Y}</b><br>Preço  US$ %{y:,.2f}<br>Retorno %{customdata[3]:+.2f}%<br>z ret  %{customdata[0]:+.2f}<extra></extra>";

  const traces = [
    { type: "scatter", mode: "lines", x: X, y: s.price, line: { color: C.price, width: 1.6 }, name: "Preço", customdata: custom, hovertemplate: hovP },
    { type: "scatter", mode: "markers", x: pick(X, iRet), y: pick(s.price, iRet), customdata: pick(custom, iRet), name: "MAD (retorno)",
      marker: { color: C.red, size: 7, line: { color: C.bg, width: 1 } }, hovertemplate: "Candidato MAD · z ret %{customdata[0]:+.2f}<extra></extra>" },
    { type: "scatter", mode: "markers", x: pick(X, iIf), y: pick(s.price, iIf), customdata: pick(custom, iIf), name: "Isolation Forest",
      marker: { color: "rgba(0,0,0,0)", size: 13, symbol: "circle-open", line: { color: C.amber, width: 1.8 } }, hovertemplate: "Isolation Forest · escore %{customdata[2]:.3f}<extra></extra>" },
    { type: "bar", x: X, y: s.volume, yaxis: "y2", name: "Volume", customdata: custom,
      marker: { color: s.vol_cand.map((v, i) => (v ? C.red : s.if_cand[i] ? "rgba(255,181,71,.75)" : C.vol)), line: { width: 0 } },
      hovertemplate: "Volume %{y:,.0f} · z vol %{customdata[1]:+.2f}<extra></extra>" },
  ];
  if (sel) {
    const i = s.date.indexOf(sel.date);
    traces.push({ type: "scatter", mode: "markers", x: [X[i]], y: [s.price[i]], hoverinfo: "skip",
      marker: { size: 18, color: "rgba(255,255,255,.12)", line: { color: "#fff", width: 2 } } });
  }

  const emTodas = new Set(D.sens.pelt_em_todas);
  const shapes = [], ann = [];
  shapes.push({ type: "line", xref: "paper", yref: "paper", x0: 0, x1: 1, y0: 0.275, y1: 0.275, line: { color: C.line, width: 1 } });
  D.pelt.forEach((d) => shapes.push({ type: "line", xref: "x", yref: "paper", x0: d, x1: d, y0: 0, y1: 1,
    line: { color: emTodas.has(d) ? C.yellow : C.cyan, width: 1.2, dash: "dash" }, opacity: 0.75, layer: "below" }));
  D.atualizacoes.forEach((a, k) => {
    shapes.push({ type: "line", xref: "x", yref: "paper", x0: noon(a.date), x1: noon(a.date), y0: 0, y1: 1, line: { color: C.green, width: 1.6 }, layer: "below" });
    ann.push({ x: noon(a.date), y: 1, xref: "x", yref: "paper", yanchor: "top", xanchor: "left", showarrow: false, xshift: 4, yshift: -2 - 16 * (k % 2),
      text: $("chart").clientWidth < 640 ? `▲ ${a.rotulo}` : `▲ ${a.rotulo} · ${dshort(a.date)}`, font: { family: MONO, size: 10.5, color: C.green }, bgcolor: "rgba(11,14,20,.85)", borderpad: 2 });
  });
  if (sel) {
    shapes.push({ type: "rect", xref: "x", yref: "paper", x0: addDays(sel.date, -1) + " 12:00", x1: addDays(sel.date, 1) + " 12:00", y0: 0, y1: 1,
      fillcolor: "rgba(255,255,255,.07)", line: { color: "rgba(255,255,255,.45)", width: 1, dash: "dot" }, layer: "below" });
    ann.push({ x: noon(sel.date), y: 0.275, xref: "x", yref: "paper", yanchor: "bottom", showarrow: false, yshift: 4,
      text: `#${sel.pos} · ${dshort(sel.date)}`, font: { family: MONO, size: 10.5, color: "#fff" }, bgcolor: "rgba(78,161,255,.85)", borderpad: 3 });
  }
  const layout = Object.assign(baseLayout(), {
    margin: { l: 8, r: 62, t: 30, b: 30 }, hovermode: "x unified", hoversubplots: "axis", hoverdistance: -1, spikedistance: -1, bargap: 0.15,
    xaxis: Object.assign({}, AX, { type: "date", anchor: "y2", showspikes: true, spikemode: "across", spikesnap: "cursor", spikecolor: C.tx3, spikethickness: 1, spikedash: "solid",
      showgrid: false, hoverformat: "%d/%m/%Y", tickformatstops: [{ dtickrange: [null, 864e5 * 25], value: "%d/%m/%y" }, { dtickrange: [864e5 * 25, null], value: "%b %Y" }], range: S.range || undefined }),
    yaxis: Object.assign({}, AX, { domain: [0.31, 1], side: "right", tickprefix: "$", tickformat: ",.0f", showspikes: true, spikemode: "across", spikecolor: C.tx3, spikethickness: 1, spikedash: "solid", fixedrange: false, title: { text: "" } }),
    yaxis2: Object.assign({}, AX, { domain: [0, 0.25], side: "right", tickformat: "~s", showspikes: false, rangemode: "tozero" }),
    shapes, annotations: ann, uirevision: S.skin + "|" + (S.rangeRev || 0),
  });
  if (S.range) Object.assign(layout.yaxis, yFit(s, S.range).y), Object.assign(layout.yaxis2, yFit(s, S.range).y2);
  const el = $("chart");
  Plotly.react(el, traces, layout, { responsive: true, displaylogo: false, scrollZoom: true, locale: "pt-BR",
    modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d", "toggleSpikelines", "hoverCompareCartesian", "hoverClosestCartesian"] });
  if (!el._bound) {
    el._bound = true;
    el.on("plotly_relayout", (ev) => {
      if (ev["xaxis.range[0]"] || ev["xaxis.range"]) {
        const r = ev["xaxis.range"] || [ev["xaxis.range[0]"], ev["xaxis.range[1]"]];
        S.range = r;
        const fit = yFit(S.data.serie, r);
        Plotly.relayout(el, { "yaxis.range": fit.y.range, "yaxis2.range": fit.y2.range });
        markRange(null);
      } else if (ev["xaxis.autorange"]) { S.range = null; markRange("0"); }
    });
    el.on("plotly_click", (ev) => {
      const p = ev.points && ev.points[0]; if (!p) return;
      const d = String(p.x).slice(0, 10);
      const e = S.data.top.find((t) => t.skin === S.skin && t.date === d);
      if (e) selecionar(e, false);
    });
  }
}
function yFit(s, r) {
  const a = Date.parse(r[0]), b = Date.parse(r[1]);
  let mn = Infinity, mx = -Infinity, vm = 0;
  s.date.forEach((d, i) => { const t = Date.parse(noon(d)); if (t >= a && t <= b) { mn = Math.min(mn, s.price[i]); mx = Math.max(mx, s.price[i]); vm = Math.max(vm, s.volume[i]); } });
  if (!Number.isFinite(mn)) return { y: {}, y2: {} };
  const pad = (mx - mn) * 0.08 || mx * 0.05;
  return { y: { range: [Math.max(0, mn - pad), mx + pad], autorange: false }, y2: { range: [0, vm * 1.08], autorange: false } };
}
function markRange(r) { document.querySelectorAll("#range-btns button").forEach((b) => b.classList.toggle("on", b.dataset.r === r)); }
function setRange(daysBack, center) {
  const s = S.data.serie, last = s.date[s.date.length - 1], first = s.date[0];
  if (center) S.range = [noon(addDays(center, -75)), noon(addDays(center, 75))];
  else if (!daysBack) S.range = null;
  else S.range = [noon(addDays(last, -daysBack)), noon(addDays(last, 1))];
  if (S.range && Date.parse(S.range[0]) < Date.parse(first)) S.range[0] = noon(first);
  S.rangeRev = (S.rangeRev || 0) + 1;
  renderChart(S.data);
}

/* ---------------- mesa de triagem ---------------- */
function renderTable(D) {
  const top = D.top;
  const of = top.filter((e) => e.comum).length;
  $("f-all").textContent = top.length; $("f-of").textContent = of; $("f-sem").textContent = top.length - of;
  document.querySelectorAll("#filtros .chip").forEach((c) => c.classList.toggle("on", c.dataset.f === S.filtro));
  const rows = top.filter((e) => S.filtro === "all" || (S.filtro === "comum" ? e.comum : !e.comum));
  const rMin = Math.min(...top.map((e) => e.integrated_rank)), rMax = Math.max(...top.map((e) => e.integrated_rank));
  const mMax = Math.max(...top.map((e) => e.robust_score)), iMax = Math.max(...top.map((e) => e.if_score));
  const cor = Object.fromEntries(D.cards.map((c) => [c.skin, c.cor]));
  const bar = (pct, color, txt) => `<div class="bar"><div class="track"><div class="fill" style="width:${Math.max(3, pct).toFixed(1)}%;background:${color}"></div></div><span class="n">${txt}</span></div>`;
  document.querySelector("#tbl-top tbody").innerHTML = rows.map((e) => {
    const rp = 15 + (85 * (e.integrated_rank - rMin)) / (rMax - rMin || 1);
    const ret = e.log_return === null ? null : 100 * e.log_return;
    return `<tr data-k="${esc(keyOf(e))}" class="${keyOf(e) === S.sel ? "sel" : ""} ${e.comum ? "" : "alert"}" tabindex="0">
      <td class="num">${e.pos}</td><td class="mono">${dbr(e.date)}</td>
      <td><span class="skin-cell" style="--rc:${cor[e.skin]}"><i></i>${esc(e.skin)}</span></td>
      <td><span class="bdg ${e.badge}" title="${esc(e.texto)}">${esc(e.rotulo)}</span></td>
      <td>${bar(rp, C.accent, f(e.integrated_rank, 4))}</td>
      <td>${bar((100 * e.robust_score) / mMax, C.red, f(e.robust_score, 2))}</td>
      <td>${bar((100 * e.if_score) / iMax, C.amber, f(e.if_score, 3))}</td>
      <td class="ctr"><span class="yn ${e.mad ? "y mad" : "n"}">${e.mad ? "SIM" : "não"}</span></td>
      <td class="ctr"><span class="yn ${e.if ? "y if" : "n"}">${e.if ? "SIM" : "não"}</span></td>
      <td class="num ${ret > 0 ? "pos" : ret < 0 ? "neg" : ""}">${ret === null ? "—" : fs(ret, 2) + "%"}</td>
      <td class="num">${usd(e.price)}</td><td class="num">${f(e.volume)}</td></tr>`;
  }).join("");
}

function renderGroups(D) {
  $("groups").innerHTML = D.grupos.map((g) => {
    const per = g.inicio === g.fim ? dbr(g.inicio) : `${dbr(g.inicio)} a ${dbr(g.fim)}`;
    const armas = g.armas.length > 1 ? g.armas.map((a) => `${a.arma} (${a.n})`).join(", ") : g.armas.map((a) => a.arma).join(", ");
    return `<div class="grp"><div class="grp-top"><span class="bdg ${g.badge}">${esc(g.rotulo)}</span><span class="grp-n">${g.n}</span></div>
      <div class="grp-p">${per}</div><div style="margin-top:4px"><b style="color:${C.tx}">${esc(armas)}</b></div><div style="margin-top:4px">${esc(g.texto)}</div></div>`;
  }).join("");
}

/* ---------------- dossiê ---------------- */
const DIAG = {
  tradeup: "O dia cai na janela de 22/10 a 02/11/2025, logo após a atualização oficial <strong>Trade Up</strong>. O índice de mercado confirma: em 23/10/2025, o retorno mediano de 1.412 skins foi de +12,7% e 72,0% delas foram marcadas. O TCC lê o alerta como <strong>choque comum do ecossistema</strong>, um contexto legítimo conhecido, e não como ação isolada sobre uma skin.",
  armory: "O dia coincide com a atualização oficial <strong>The Armory</strong> (02/10/2024; aparece em 03/10 porque as séries estão em UTC). O índice de mercado caiu 4,2% e 14,3% do universo foi marcado: <strong>choque comum</strong> com contexto legítimo conhecido.",
  liquidez: "Não há atualização oficial, e o índice de mercado ficou estável: é <strong>movimento próprio do item</strong>. A hipótese registrada no TCC é <strong>problema de dado ou venda isolada</strong> num dia de <strong>baixa liquidez</strong>: com poucas unidades negociadas e um único ponto de preço na fonte, uma venda fora do padrão desloca o preço diário.",
  semexp: "O movimento não foi próprio do item: em 01/02/2025 o índice de mercado caiu 11,7% e 46,7% do universo foi marcado, depois de uma alta de 19,7% entre 28 e 31/01 com volume abaixo do habitual. O período coincide com a atualização <strong>Premier Season Two</strong> (29/01/2025) e com relatos de usuários sobre falha na criação de anúncios. É um <strong>choque comum com causa não confirmada</strong>.",
  nc: "Este evento entrou no Top 25 com os parâmetros atuais, mas não faz parte da tabela do TCC. A abrangência é calculada pela amplitude do dia; a causa exige investigação contextual.",
};
function renderDossie(D) {
  const el = $("dossie");
  const e = D.top.find((t) => keyOf(t) === S.sel);
  if (!e) {
    const c = D.cards.find((x) => x.skin === S.skin);
    el.style.setProperty("--rc", c.cor);
    el.innerHTML = `<div class="ds-kicker"><span>Dossiê do evento</span></div><div class="ds-title">${esc(S.skin)}</div>
      <div class="ds-diag">Nenhuma observação desta skin está entre as 25 de maior prioridade do ranking integrado. Os alertas MAD (${c.alertas_mad}) e IF (${c.alertas_if}) aparecem no gráfico; selecione uma linha da mesa de triagem para abrir um dossiê.</div>`;
    return;
  }
  const c = D.cards.find((x) => x.skin === e.skin);
  el.style.setProperty("--rc", c.cor);
  const varD1 = e.preco_anterior ? 100 * (e.price / e.preco_anterior - 1) : null;
  const ret = e.log_return === null ? null : 100 * e.log_return;
  const xVol = e.volume / e.volume_mediano_skin;
  const domina = Math.abs(e.z_volume ?? 0) >= Math.abs(e.z_return ?? 0) ? "volume" : "retorno";
  const lim = S.limiar;
  const valve = D.atualizacoes.map((a) => ({ ...a, d: dias(e.date, a.date) })).sort((a, b) => Math.abs(a.d) - Math.abs(b.d))[0];
  const pelt = e.quebra_proxima ? dias(e.date, e.quebra_proxima.data) : null;
  const rel = (d) => (d === 0 ? "D0" : d > 0 ? `D+${d}` : `D−${-d}`);
  const i = D.top.indexOf(e);
  el.innerHTML = `
    <div class="ds-kicker"><span>Dossiê do evento · #${e.pos} de 25</span><span>${e.skin === S.skin ? "no gráfico" : ""}</span></div>
    <div><div class="ds-title">${esc(e.skin)}<span class="rar" style="--rc:${c.cor};color:${c.cor};font-size:10px">${c.raridade}</span></div>
      <div class="ds-date">${dbr(e.date)} · <span class="bdg ${e.badge}">${esc(e.rotulo)}</span></div></div>
    <div class="ds-grid">
      <div class="ds-cell"><div class="l">Preço mediano</div><div class="v">${usd(e.price)}</div><div class="s ${varD1 > 0 ? "pos" : "neg"}">${fs(varD1, 2)}% vs. D−1</div></div>
      <div class="ds-cell"><div class="l">Volume</div><div class="v">${f(e.volume)} un.</div><div class="s">${f(xVol, 1)}× a mediana (${f(e.volume_mediano_skin)})</div></div>
      <div class="ds-cell"><div class="l">z robusto · retorno</div><div class="v ${Math.abs(e.z_return) >= lim ? "neg" : ""}">${fs(e.z_return, 2)}</div><div class="s">ret. log ${fs(ret, 2)}%</div></div>
      <div class="ds-cell"><div class="l">z robusto · volume</div><div class="v ${Math.abs(e.z_volume) >= lim ? "neg" : ""}">${fs(e.z_volume, 2)}</div><div class="s">limiar |z| ≥ ${f(lim, 1)}</div></div>
    </div>
    <div class="ds-flags">
      <span class="flag mad ${e.mad ? "on" : ""}">MAD ${e.mad ? "✓ candidato" : "× abaixo do limiar"}</span>
      <span class="flag if ${e.if ? "on" : ""}">IF ${e.if ? "✓ anômalo" : "× normal"}</span>
      <span class="flag ctx">${rel(valve.d)} do ${esc(valve.rotulo)}</span>
      <span class="flag ctx" title="Percentual das 1.412 skins do universo marcadas pelo baseline no dia; limite atual ${f(S.amplitude)}%">Amplitude ${esc(e.amp_txt)} · ${e.comum ? "choque comum" : "próprio do item"}</span>
      ${pelt !== null ? `<span class="flag ctx" title="Quebra de regime mais próxima (penalização ${S.penalidade})">PELT ${pelt === 0 ? "no dia" : (pelt > 0 ? pelt + "d após" : -pelt + "d antes")} da quebra ${dshort(e.quebra_proxima.data)}</span>` : ""}
    </div>
    <div class="ds-ranks">
      ${rankRow("Posto MAD", e.robust_rank, C.red)}${rankRow("Posto IF", e.if_rank, C.amber)}${rankRow("Rₜ integrado", e.integrated_rank, C.accent)}
    </div>
    <div class="ds-formula">escore MAD = max(|z<sub>ret</sub>|, |z<sub>vol</sub>|) = ${f(e.robust_score, 2)} <span style="color:${C.tx2}">(${domina})</span><br>
      escore IF = ${f(e.if_score, 4)}<br>R<sub>t</sub> = (${f(e.robust_rank, 4)} + ${f(e.if_rank, 4)}) / 2 = <b style="color:${C.tx}">${f(e.integrated_rank, 4)}</b></div>
    <div class="ds-diag"><div class="h">Diagnóstico do TCC</div><p>${DIAG[e.badge]}</p><p style="color:${C.tx3};font-size:11.5px">Registro: ${esc(e.texto)}.</p></div>
    <div class="ds-nav"><button class="btn" id="ds-prev" ${i <= 0 ? "disabled" : ""}>${i > 0 ? "← #" + D.top[i - 1].pos : "início"}</button><button class="btn" id="ds-next" ${i >= D.top.length - 1 ? "disabled" : ""}>${i < D.top.length - 1 ? "#" + D.top[i + 1].pos + " →" : "fim"}</button></div>`;
  $("ds-prev").onclick = () => i > 0 && selecionar(D.top[i - 1], true);
  $("ds-next").onclick = () => i < D.top.length - 1 && selecionar(D.top[i + 1], true);
}
const rankRow = (l, v, c) => `<div class="ds-rank"><span>${l}</span><div class="track"><div class="fill" style="width:${(100 * v).toFixed(2)}%;background:${c}"></div></div><span class="n">${f(v, 4)}</span></div>`;

function selecionar(e, zoom) {
  S.sel = keyOf(e);
  const mudouSkin = e.skin !== S.skin;
  if (zoom || mudouSkin) S.range = [noon(addDays(e.date, -75)), noon(addDays(e.date, 75))], S.rangeRev = (S.rangeRev || 0) + 1;
  if (mudouSkin) { S.skin = e.skin; carregar(); }
  else { renderChart(S.data); renderTable(S.data); renderDossie(S.data); markRange(null); }
}

/* ---------------- estudo de evento ---------------- */
function renderEvento(D) {
  const R = D.evento.resumo, cor = Object.fromEntries(D.cards.map((c) => [c.skin, c])), ev = D.evento.data;
  const glock = R.find((r) => r.item.startsWith("Glock")), m4 = R.find((r) => r.item.startsWith("M4A1-S"));
  $("contrast").innerHTML = `
    <div class="ct" style="--rc:${cor[glock.item].cor}"><span class="rar" style="--rc:${cor[glock.item].cor}">${cor[glock.item].raridade}</span><div class="ct-name">${esc(glock.item)}</div>
      <div class="ct-big pos">${fs(glock.retorno_30d, 1)}%</div><div class="ct-sub">D+30 · ${usd(glock.preco_base)} → ${usd(glock.preco_30d)} · D+7 ${fs(glock.retorno_7d, 1)}%</div></div>
    <div class="ct" style="--rc:${cor[m4.item].cor}"><span class="rar" style="--rc:${cor[m4.item].cor}">${cor[m4.item].raridade}</span><div class="ct-name">${esc(m4.item)}</div>
      <div class="ct-big neg">${fs(m4.retorno_7d, 1)}%</div><div class="ct-sub">D+7 · ${usd(m4.preco_base)} → ${usd(m4.preco_7d)} · D+30 ${fs(m4.retorno_30d, 1)}%</div></div>
    <div class="ct" style="--rc:${C.green}"><div class="ct-text">O mesmo choque produziu <b>direções opostas</b>: a única skin <b>Classified</b> da cesta quase triplicou em 30 dias, enquanto a M4A1-S, <b>Covert</b>, caiu ${f(Math.abs(m4.retorno_7d), 1)}% na primeira semana e ainda fechou D+30 abaixo de D−1. Já o volume subiu em <b>todas</b> as cinco skins (razão de ${f(Math.min(...R.map((r) => r.razao_volume_mediano)), 1)}× a ${f(Math.max(...R.map((r) => r.razao_volume_mediano)), 1)}×). É por isso que o TCC classifica os 20 alertas dessa janela como contexto legítimo, e não como manipulação.</div></div>`;

  const curvas = D.evento.curvas;
  const tr = Object.entries(curvas).map(([sk, c]) => ({ type: "scatter", mode: "lines", name: arma(sk), x: c.date.map(noon), y: c.index,
    line: { color: SKIN_COR[sk], width: sk.startsWith("Glock") || sk.startsWith("M4A1") ? 2.4 : 1.4 }, opacity: sk.startsWith("Glock") || sk.startsWith("M4A1") ? 1 : 0.7,
    hovertemplate: `${arma(sk)} %{y:.1f}<extra></extra>` }));
  const marks = [[7, "D+7"], [30, "D+30"]].map(([n, l]) => addDays(ev, n));
  Plotly.react("chart-evento", tr, Object.assign(baseLayout(), {
    showlegend: true, legend: { orientation: "h", y: 1.12, x: 0, font: { size: 11, color: C.tx2 } }, margin: { l: 44, r: 12, t: 30, b: 30 }, hovermode: "x unified",
    xaxis: Object.assign({}, AX, { type: "date", tickformat: "%d/%m", hoverformat: "%d/%m/%Y", showgrid: false }),
    yaxis: Object.assign({}, AX, { title: { text: "" }, ticksuffix: "" }),
    shapes: [
      { type: "line", xref: "x", yref: "paper", x0: noon(ev), x1: noon(ev), y0: 0, y1: 1, line: { color: C.green, width: 1.6, dash: "dash" } },
      { type: "line", xref: "paper", yref: "y", x0: 0, x1: 1, y0: 100, y1: 100, line: { color: C.tx3, width: 1, dash: "dot" } },
      ...marks.map((m) => ({ type: "line", xref: "x", yref: "paper", x0: noon(m), x1: noon(m), y0: 0, y1: 1, line: { color: C.line2, width: 1 } })),
    ],
    annotations: [{ x: noon(ev), y: 1, xref: "x", yref: "paper", yanchor: "top", xanchor: "right", xshift: -4, showarrow: false, text: "Trade Up · D0", font: { family: MONO, size: 10.5, color: C.green } },
      ...marks.map((m, i) => ({ x: noon(m), y: 1, xref: "x", yref: "paper", yanchor: "top", xanchor: "left", xshift: 3, showarrow: false, text: ["D+7", "D+30"][i], font: { family: MONO, size: 10, color: C.tx3 } }))],
  }), { responsive: true, displaylogo: false, displayModeBar: false });

  const ordem = [...R].reverse();
  const hz = [["retorno_d0", "D0", "#5b6b8c"], ["retorno_7d", "D+7", "#4ea1ff"], ["retorno_30d", "D+30", "#c9d6ff"]];
  Plotly.react("chart-horiz", hz.map(([k, l, col]) => ({ type: "bar", orientation: "h", name: l, y: ordem.map((r) => arma(r.item)), x: ordem.map((r) => r[k]),
    marker: { color: col }, text: ordem.map((r) => fs(r[k], 1) + "%"), textposition: "outside", cliponaxis: false, textfont: { family: MONO, size: 10, color: C.tx2 },
    hovertemplate: `%{y} · ${l}: %{x:+.1f}%<extra></extra>` })), Object.assign(baseLayout(), {
    barmode: "group", bargap: 0.25, bargroupgap: 0.05, showlegend: true, legend: { orientation: "h", y: 1.12, x: 0, font: { size: 11, color: C.tx2 } },
    margin: { l: 100, r: 50, t: 30, b: 30 },
    xaxis: Object.assign({}, AX, { ticksuffix: "%", zeroline: true, zerolinecolor: C.tx3, range: [-62, 232] }),
    yaxis: Object.assign({}, AX, { tickfont: { family: UI, size: 11.5, color: C.tx }, showgrid: false, ticklabelstandoff: 8 }),
  }), { responsive: true, displaylogo: false, displayModeBar: false });

  const cls = (v) => (v > 0 ? "pos" : v < 0 ? "neg" : "");
  document.querySelector("#tbl-evento tbody").innerHTML = R.map((r) => `<tr>
    <td><span class="skin-cell" style="--rc:${cor[r.item].cor}"><i></i>${esc(r.item)}</span></td><td style="color:${cor[r.item].cor};font-size:11px;font-weight:600">${cor[r.item].raridade}</td>
    <td class="num">${usd(r.preco_base)}</td><td class="num ${cls(r.retorno_d0)}">${fs(r.retorno_d0, 1)}%</td><td class="num ${cls(r.retorno_7d)}">${fs(r.retorno_7d, 1)}%</td>
    <td class="num ${cls(r.retorno_30d)}">${fs(r.retorno_30d, 1)}%</td><td class="num">${f(r.volume_mediano_antes, 1)}</td><td class="num">${f(r.volume_mediano_depois, 1)}</td>
    <td class="num"><b>${f(r.razao_volume_mediano, 1)}×</b></td><td class="num">${f(r.volume_maximo_antes)}</td><td class="num">${f(r.volume_maximo_depois)}</td></tr>`).join("");
  const d0 = R.map((r) => r.retorno_d0), a7 = R.filter((r) => r.retorno_7d > 0).map((r) => r.retorno_7d), d30 = R.map((r) => r.retorno_30d), rz = R.map((r) => r.razao_volume_mediano);
  $("ev-bullets").innerHTML = `
    <li>No próprio dia 22/10, os preços variaram de <b>${fs(Math.min(...d0), 1)}%</b> a <b>${fs(Math.max(...d0), 1)}%</b> em relação a D−1: a reação de preço não foi imediata.</li>
    <li>Até D+7, ${a7.length} skins acumularam alta de <b>${fs(Math.min(...a7), 1)}%</b> a <b>${fs(Math.max(...a7), 1)}%</b>; até D+30, a variação foi de <b>${fs(Math.min(...d30), 1)}%</b> a <b>${fs(Math.max(...d30), 1)}%</b>.</li>
    <li>O volume mediano (D−30…D−1 contra D0…D+30) aumentou de <b>${f(Math.min(...rz), 1)}</b> a <b>${f(Math.max(...rz), 1)} vezes</b>.</li>
    <li>A simultaneidade em skins diferentes favorece a hipótese de <b style="font-family:inherit">choque comum do ecossistema</b>; a janela é descritiva e não isola o efeito causal da atualização.</li>`;
}

/* ---------------- sensibilidade ---------------- */
function renderSens(D) {
  const sm = D.sens.mad_fina, tcc = new Set(D.sens.grade_mad.map((v) => v.toFixed(1)));
  const curLine = (x, txt) => ({ shapes: [{ type: "line", xref: "x", yref: "paper", x0: x, x1: x, y0: 0, y1: 1, line: { color: C.amber, width: 1.4, dash: "dash" } }],
    annotations: [{ x, y: 1, xref: "x", yref: "paper", yanchor: "bottom", showarrow: false, text: txt, font: { family: MONO, size: 10.5, color: C.amber } }] });
  const ptsT = sm.filter((r) => tcc.has(r.limiar.toFixed(1)));
  Plotly.react("chart-sens-mad", [
    { type: "scatter", mode: "lines", x: sm.map((r) => r.limiar), y: sm.map((r) => r.eventos), line: { color: C.red, width: 2, shape: "hv" }, fill: "tozeroy", fillcolor: "rgba(255,90,90,.08)", hovertemplate: "limiar %{x:.1f} · %{y} eventos<extra></extra>" },
    { type: "scatter", mode: "markers", x: ptsT.map((r) => r.limiar), y: ptsT.map((r) => r.eventos), marker: { color: C.red, size: 9, line: { color: C.bg2, width: 2 } }, hovertemplate: "TCC · limiar %{x:.1f} · %{y} eventos<extra></extra>" },
  ], Object.assign(baseLayout(), { margin: { l: 40, r: 10, t: 20, b: 30 }, hovermode: "closest",
    xaxis: Object.assign({}, AX, { dtick: 0.5, showgrid: false }), yaxis: Object.assign({}, AX, { rangemode: "tozero" }), ...curLine(S.limiar, "atual " + f(S.limiar, 1)) }),
  { responsive: true, displaylogo: false, displayModeBar: false });
  document.querySelector("#tbl-sens-mad tbody").innerHTML = D.sens.mad_grade.map((r) => {
    const cur = Math.abs(r.limiar - S.limiar) < 1e-9, t = tcc.has(r.limiar.toFixed(1));
    return `<tr class="${cur ? "cur" : ""} ${t ? "tcc" : ""}"><td class="num">${f(r.limiar, 1)}</td><td class="num">${f(r.eventos)}</td><td class="num">${f(r.percentual, 2)}%</td><td>${cur ? "atual" : ""}</td></tr>`;
  }).join("");

  // PELT: penalização × datas
  $("sens-pelt-skin").textContent = S.skin;
  const P = D.sens.pelt, todas = new Set(D.sens.pelt_em_todas), tccP = new Set(D.sens.grade_pelt);
  const xs = [], ys = [], cs = [], tx = [];
  P.forEach((r) => r.datas.forEach((d) => { xs.push(noon(d)); ys.push(r.penalizacao); cs.push(todas.has(d) ? C.yellow : C.cyan); tx.push(dbr(d)); }));
  Plotly.react("chart-sens-pelt", [
    { type: "scatter", mode: "markers", x: xs, y: ys, text: tx, marker: { symbol: "line-ns-open", size: 13, color: cs, line: { width: 2.2, color: cs } }, hovertemplate: "pen. %{y} · %{text}<extra></extra>" },
    { type: "scatter", mode: "text", x: P.map(() => D.serie.date[D.serie.date.length - 1] + " 12:00"), y: P.map((r) => r.penalizacao), text: P.map((r) => `${r.datas.length}`),
      textposition: "middle right", textfont: { family: MONO, size: 10, color: C.tx3 }, hoverinfo: "skip", cliponaxis: false },
  ], Object.assign(baseLayout(), { margin: { l: 44, r: 26, t: 10, b: 30 }, hovermode: "closest",
    xaxis: Object.assign({}, AX, { type: "date", tickformat: "%b %y", showgrid: false, range: [noon(D.serie.date[0]), noon(D.serie.date[D.serie.date.length - 1])] }),
    yaxis: Object.assign({}, AX, { autorange: "reversed", tickvals: P.map((r) => r.penalizacao),
      ticktext: P.map((r) => (r.penalizacao === S.penalidade ? `▶ ${r.penalizacao}` : tccP.has(r.penalizacao) ? `${r.penalizacao}•` : `${r.penalizacao}`)), showgrid: false, title: { text: "" } }),
    shapes: [{ type: "rect", xref: "paper", yref: "y", x0: 0, x1: 1, y0: S.penalidade - 0.45, y1: S.penalidade + 0.45, fillcolor: "rgba(255,181,71,.10)", line: { width: 0 }, layer: "below" },
      ...D.atualizacoes.map((a) => ({ type: "line", xref: "x", yref: "paper", x0: noon(a.date), x1: noon(a.date), y0: 0, y1: 1, line: { color: C.green, width: 1 }, layer: "below" }))],
  }), { responsive: true, displaylogo: false, displayModeBar: false });

  // IF × baseline
  const I = D.sens.if;
  const conc = I.map((r) => (r.eventos_if ? (100 * r.ambos) / r.eventos_if : null));
  Plotly.react("chart-sens-if", [
    { type: "bar", name: "Eventos IF", x: I.map((r) => r.contaminacao), y: I.map((r) => r.eventos_if), marker: { color: "rgba(255,181,71,.75)" }, hovertemplate: "c=%{x:.3f} · %{y} eventos IF<extra></extra>" },
    { type: "bar", name: "Também MAD", x: I.map((r) => r.contaminacao), y: I.map((r) => r.ambos), marker: { color: C.red }, hovertemplate: "c=%{x:.3f} · %{y} também MAD<extra></extra>" },
    { type: "scatter", mode: "lines+markers", name: "Concordância %", x: I.map((r) => r.contaminacao), y: conc, yaxis: "y2", line: { color: C.tx, width: 1.6 }, marker: { size: 5 }, hovertemplate: "concordância %{y:.0f}%<extra></extra>" },
  ], Object.assign(baseLayout(), { barmode: "group", bargap: 0.2, showlegend: true, legend: { orientation: "h", y: -0.2, x: 0, font: { size: 10.5, color: C.tx2 } },
    margin: { l: 40, r: 40, t: 20, b: 54 }, hovermode: "x unified",
    xaxis: Object.assign({}, AX, { tickvals: I.map((r) => r.contaminacao), ticktext: I.map((r) => f(r.contaminacao, 3)), showgrid: false }),
    yaxis: Object.assign({}, AX, { rangemode: "tozero" }),
    yaxis2: Object.assign({}, AX, { overlaying: "y", side: "right", range: [0, 105], ticksuffix: "%", showgrid: false }),
    ...curLine(S.contaminacao, "atual " + f(S.contaminacao, 3)) }), { responsive: true, displaylogo: false, displayModeBar: false });
  document.querySelector("#tbl-sens-if tbody").innerHTML = I.map((r, i) => {
    const cur = Math.abs(r.contaminacao - S.contaminacao) < 1e-9, t = Math.abs(r.contaminacao - 0.02) < 1e-9;
    return `<tr class="${cur ? "cur" : ""} ${t ? "tcc" : ""}"><td class="num">${f(r.contaminacao, 3)}</td><td class="num">${f(r.eventos_if)}</td><td class="num">${f(r.ambos)}</td><td class="num">${f(conc[i])}%</td></tr>`;
  }).join("");
}

/* ---------------- mercado, regra e piso de detecção ---------------- */
function renderMercado(D) {
  const M = D.mercado, R = D.regra, r = R.resumo, s = M.serie;
  const cor = Object.fromEntries(D.cards.map((c) => [c.skin, c.cor]));
  $("kpis-mercado").innerHTML = [
    `<div class="kpi"><div class="kpi-l">Universo do índice</div><div class="kpi-v">${f(M.universo.skins)}</div><div class="kpi-d">skins · ${f(M.universo.taxa_candidatos, 2)}% de candidatos</div></div>`,
    `<div class="kpi"><div class="kpi-l">Dias de choque comum</div><div class="kpi-v" style="color:${C.red}">${f(M.dias_comuns)}</div><div class="kpi-d">de ${f(M.dias)} · amplitude &gt; ${f(M.limite)}%</div></div>`,
    `<div class="kpi"><div class="kpi-l">Amplitude típica</div><div class="kpi-v">${f(M.amp_mediana, 2)}%</div><div class="kpi-d">mediana diária do universo</div></div>`,
    `<div class="kpi"><div class="kpi-l">Prioridades em choque comum</div><div class="kpi-v" style="color:${C.amber}">${M.top_comum} de ${M.top_total}</div><div class="kpi-d">${M.top_total - M.top_comum} próprias do item</div></div>`,
    `<div class="kpi"><div class="kpi-l">Episódios</div><div class="kpi-v">${r.total}</div><div class="kpi-d">${f(r.item_dias)} item-dias · ${r.associados} com evento</div></div>`,
    `<div class="kpi"><div class="kpi-l">Atendem à regra</div><div class="kpi-v" style="color:${r.nivel3 ? C.red : C.green}">${r.nivel3}</div><div class="kpi-d">das quatro condições</div></div>`,
  ].join("");

  // gráfico: índice (acima) e amplitude (abaixo)
  const X = s.date.map(noon), lim = M.limite;
  const custom = s.date.map((_, i) => [s.ret[i], s.z[i], s.amp[i]]);
  const shapes = [{ type: "line", xref: "paper", yref: "y2", x0: 0, x1: 1, y0: lim, y1: lim, line: { color: C.amber, width: 1.2, dash: "dash" } }], ann = [];
  let k = 0;
  M.calendario.forEach((e) => {
    const up = e.tipo === "atualização";
    shapes.push({ type: "line", xref: "x", yref: "paper", x0: noon(e.date), x1: noon(e.date), y0: 0, y1: 1, layer: "below",
      line: { color: up ? C.green : C.accent, width: up ? 1.5 : 1, dash: up ? "solid" : "dot" }, opacity: up ? 0.9 : 0.55 });
    if (up) ann.push({ x: noon(e.date), y: 1, xref: "x", yref: "paper", yanchor: "top", xanchor: "left", showarrow: false, xshift: 4, yshift: -2 - 16 * (k++ % 2),
      text: `▲ ${e.rotulo}`, font: { family: MONO, size: 10.5, color: C.green }, bgcolor: "rgba(11,14,20,.85)", borderpad: 2 });
  });
  ann.push({ x: 0, y: lim, xref: "paper", yref: "y2", xanchor: "left", yanchor: "bottom", showarrow: false, xshift: 4,
    text: `limite ${f(lim)}%`, font: { family: MONO, size: 10, color: C.amber } });
  Plotly.react("chart-mercado", [
    { type: "scatter", mode: "lines", x: X, y: s.nivel, line: { color: C.price, width: 1.6 }, customdata: custom,
      hovertemplate: "<b>%{x|%d/%m/%Y}</b><br>Índice %{y:,.1f}<br>Retorno %{customdata[0]:+.1f}% · escore %{customdata[1]:+.1f}<extra></extra>" },
    { type: "bar", x: X, y: s.amp, yaxis: "y2", marker: { color: s.amp.map((v) => (v > lim ? C.red : C.vol)), line: { width: 0 } },
      hovertemplate: "Amplitude %{y:.1f}% do universo<extra></extra>" },
  ], Object.assign(baseLayout(), {
    margin: { l: 8, r: 56, t: 34, b: 30 }, hovermode: "x unified", bargap: 0.1,
    xaxis: Object.assign({}, AX, { type: "date", anchor: "y2", showgrid: false, hoverformat: "%d/%m/%Y", tickformat: "%b %Y" }),
    yaxis: Object.assign({}, AX, { domain: [0.42, 1], side: "right", tickformat: ",.0f" }),
    yaxis2: Object.assign({}, AX, { domain: [0, 0.34], side: "right", ticksuffix: "%", rangemode: "tozero" }),
    shapes, annotations: ann,
  }), { responsive: true, displaylogo: false, displayModeBar: false });

  // regra: condições, filtro e tabela de episódios
  $("regra-n").textContent = r.total;
  const cond = (l, n, d, cls = "") => `<div class="cond ${cls}"><div class="l">${l}</div><div class="n">${n}</div><div class="d">${d}</div></div>`;
  $("conds").innerHTML = [
    cond("1 · Movimento próprio", r.c1, `nenhum dia com amplitude acima de ${f(M.limite)}%`),
    cond("2 · Alta conjunta", r.c2, `z de retorno ≥ ${f(S.limiar, 1)} e z de volume ≥ ${f(S.volz, 1)}`),
    cond("3 · Reversão", r.c3, `devolve metade da alta em até ${S.reversao} dias`),
    cond("4 · Sem evento público", r.c4, "nenhum evento do calendário associado"),
    cond("Quatro condições", r.nivel3, r.nivel3 ? "possível sinal: exige conferência com outra fonte" : `nenhum episódio; condições 1 e 2 juntas: ${r.c1c2}`, "final" + (r.nivel3 ? " hit" : "")),
  ].join("");
  $("fe-all").textContent = r.total; $("fe-prop").textContent = r.c1c4; $("fe-n3").textContent = r.nivel3;
  document.querySelectorAll("#filtros-ep .chip").forEach((c) => c.classList.toggle("on", c.dataset.f === S.filtroEp));
  const ck = (v) => `<span class="ck ${v ? "y" : "n"}">${v ? "✓" : "×"}</span>`;
  const eps = R.episodios.filter((e) => S.filtroEp === "all" || (S.filtroEp === "proprio" ? e.c1 && e.c4 : e.nivel3));
  document.querySelector("#tbl-ep tbody").innerHTML = eps.length ? eps.map((e) => `<tr class="${e.nivel3 ? "hit" : ""}">
    <td><span class="skin-cell" style="--rc:${cor[e.skin]}"><i></i>${esc(e.skin)}</span></td>
    <td class="mono">${e.inicio === e.fim ? dbr(e.inicio) : dshort(e.inicio) + " a " + dshort(e.fim)}</td><td class="num">${e.dias}</td>
    <td class="num ${e.amp_max > M.limite ? "neg" : ""}">${f(e.amp_max, 1)}%</td><td class="num">${fs(e.z_ret_max, 1)}</td><td class="num">${fs(e.z_vol_max, 1)}</td>
    <td class="num ${e.alta > 0 ? "pos" : e.alta < 0 ? "neg" : ""}">${e.alta === null ? "—" : fs(e.alta, 1) + "%"}</td>
    <td class="ctr">${ck(e.c1)}</td><td class="ctr">${ck(e.c2)}</td><td class="ctr">${ck(e.c3)}</td><td class="ctr">${ck(e.c4)}</td>
    <td>${e.evento ? esc(e.evento) : `<span style="color:${C.tx3}">—</span>`}</td>
    <td class="ctr"><span class="res ${e.nivel3 ? "hit" : ""}">${e.nivel3 ? "ATENDE" : "não atende"}</span></td></tr>`).join("")
    : `<tr><td colspan="13" style="text-align:center;color:${C.tx3};padding:18px">Nenhum episódio neste filtro.</td></tr>`;

  // calendário × índice
  const cls = (v) => (v > 0 ? "pos" : v < 0 ? "neg" : "");
  document.querySelector("#tbl-cal tbody").innerHTML = M.calendario.map((e) => `<tr>
    <td class="mono">${dshort(e.date)}</td><td><span class="tipo-tag" style="color:${e.tipo === "atualização" ? C.green : C.accent}">${e.tipo === "atualização" ? "atual." : "promo"}</span>${esc(e.evento.replace(/^Atualização (do )?|^Promoção de /, ""))}</td>
    <td class="num ${cls(e.d0)}">${fs(e.d0, 1)}%</td><td class="num ${cls(e.d1)}">${fs(e.d1, 1)}%</td>
    <td class="num">${e.choque ? "<b>" + fs(e.escore, 1) + "</b>" : fs(e.escore, 1)}</td><td class="num ${e.amp_max > M.limite ? "neg" : ""}">${f(e.amp_max, 1)}%</td></tr>`).join("");

  $("cal-note").innerHTML = `Taxa-base: <b>${f(M.pct_dias_evento, 1)}%</b> dos dias da janela ficam a até 7 dias depois de algum evento; <b>${r.associados} de ${r.total}</b> episódios estão associados a um evento. Escore em negrito: choque do índice (|z| ≥ 4) em D0 ou D+1.`;

  // baseline original × ajustado
  document.querySelector("#tbl-adj tbody").innerHTML = M.ajustado.map((a) => `<tr class="${a.skin === "Total" ? "sep" : ""}">
    <td>${a.skin === "Total" ? "<b>Total</b>" : `<span class="skin-cell" style="--rc:${cor[a.skin]}"><i></i>${esc(arma(a.skin))}</span>`}</td>
    <td class="num">${a.original}</td><td class="num">${a.ajustado}</td><td class="num">${a.comum}</td><td class="num">${a.ep_original}</td><td class="num">${a.ep_ajustado}</td></tr>`).join("");

  // sensibilidade da regra
  document.querySelector("#tbl-sens-regra tbody").innerHTML = R.grade_volz.map((v) => `<tr><td class="num">${f(v, 1)}</td>${R.grade_rev.map((p) => {
    const n = R.sens.find((x) => x.volz === v && x.reversao === p).n, cur = v === S.volz && p === S.reversao;
    return `<td class="num ${cur ? "curcell" : ""}" title="${cur ? "configuração atual" : ""}">${n}</td>`;
  }).join("")}</tr>`).join("");

  // piso de detecção (picos de um dia)
  $("th-sint-skin").textContent = arma(S.skin);
  const nome = { preco: "Preço", volume: "Volume", preco_e_volume: "Conjunto" };
  const pico = (x) => (x.tipo === "preco" ? `+${f(x.preco)}%` : x.tipo === "volume" ? `×${f(x.volume)}` : `+${f(x.preco)}% e ×${f(x.volume)}`);
  const heat = (v) => `<td class="num heat" style="--h:${v}%">${f(v, 1)}</td>`;
  let ant = null;
  document.querySelector("#tbl-sint tbody").innerHTML = D.sinteticas.filter((x) => x.duracao === 1).map((x) => {
    const sep = ant !== null && ant !== x.tipo; ant = x.tipo;
    return `<tr class="${sep ? "sep" : ""}"><td><span class="tipo-tag">${nome[x.tipo]}</span><span class="mono">${pico(x)}</span></td>${heat(x.global)}${heat(x.movel90)}${heat(x.if)}${heat(x.por_skin[S.skin])}</tr>`;
  }).join("");
}

/* ---------------- controles ---------------- */
let deb;
function fill(sl) { const p = (100 * (sl.value - sl.min)) / (sl.max - sl.min); sl.style.setProperty("--fill", p + "%"); }
function syncSliders() {
  const L = $("sl-limiar"), P = $("sl-pen"), Ct = $("sl-cont");
  L.value = S.limiar; P.value = S.penalidade; Ct.value = S.contaminacao;
  const A = $("sl-amp"), V = $("sl-volz"), R = $("sl-rev");
  A.value = S.amplitude; V.value = S.volz; R.value = S.reversao;
  [L, P, Ct, A, V, R].forEach(fill);
  $("out-limiar").textContent = f(S.limiar, 1); $("out-pen").textContent = S.penalidade; $("out-cont").textContent = f(S.contaminacao, 3);
  $("out-amp").textContent = f(S.amplitude) + "%"; $("out-volz").textContent = f(S.volz, 1); $("out-rev").textContent = S.reversao + " dias";
}
function bind() {
  const on = (id, fn) => $(id).addEventListener("input", (ev) => { fn(+ev.target.value); syncSliders(); renderMode(); clearTimeout(deb); deb = setTimeout(carregar, 90); });
  on("sl-limiar", (v) => (S.limiar = Math.round(v * 10) / 10));
  on("sl-pen", (v) => (S.penalidade = Math.round(v)));
  on("sl-cont", (v) => (S.contaminacao = Math.round(v * 1000) / 1000));
  on("sl-amp", (v) => (S.amplitude = Math.round(v)));
  on("sl-volz", (v) => (S.volz = Math.round(v * 2) / 2));
  on("sl-rev", (v) => (S.reversao = Math.round(v)));
  $("filtros-ep").addEventListener("click", (ev) => { const c = ev.target.closest(".chip"); if (!c) return; S.filtroEp = c.dataset.f; renderMercado(S.data); });
  $("btn-reset").addEventListener("click", () => { Object.assign(S, PADRAO); syncSliders(); renderMode(); carregar(); });
  $("cards").addEventListener("click", (ev) => {
    const b = ev.target.closest(".card"); if (!b || b.dataset.skin === S.skin) return;
    S.skin = b.dataset.skin; S.range = null; S.rangeRev = (S.rangeRev || 0) + 1; markRange("0");
    const first = S.data.top.find((e) => e.skin === S.skin); S.sel = first ? keyOf(first) : null;
    carregar();
  });
  const tb = document.querySelector("#tbl-top tbody");
  const pickRow = (tr) => { const e = S.data.top.find((t) => keyOf(t) === tr.dataset.k); if (e) selecionar(e, true); };
  tb.addEventListener("click", (ev) => { const tr = ev.target.closest("tr"); if (tr) pickRow(tr); });
  tb.addEventListener("keydown", (ev) => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); pickRow(ev.target.closest("tr")); } });
  $("filtros").addEventListener("click", (ev) => { const c = ev.target.closest(".chip"); if (!c) return; S.filtro = c.dataset.f; renderTable(S.data); });
  $("range-btns").addEventListener("click", (ev) => { const b = ev.target.closest("button"); if (!b) return; markRange(b.dataset.r); setRange(+b.dataset.r); });
  document.querySelectorAll(".tab").forEach((t) => t.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((x) => { x.classList.toggle("active", x === t); x.setAttribute("aria-selected", x === t); });
    document.querySelectorAll(".tabpane").forEach((p) => p.classList.toggle("active", p.id === "tab-" + t.dataset.tab));
    window.dispatchEvent(new Event("resize"));
  }));
}

window.addEventListener("DOMContentLoaded", () => {
  if (!window.Plotly) { setStatus("err", "Plotly.js não carregou (sem internet?)"); return; }
  Plotly.register({ moduleType: "locale", name: "pt-BR", dictionary: {}, format: {
    days: ["Domingo", "Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado"], shortDays: ["Dom", "Seg", "Ter", "Qua", "Qui", "Sex", "Sáb"],
    months: ["Janeiro", "Fevereiro", "Março", "Abril", "Maio", "Junho", "Julho", "Agosto", "Setembro", "Outubro", "Novembro", "Dezembro"],
    shortMonths: ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"], decimal: ",", thousands: ".", date: "%d/%m/%Y" } });
  Plotly.setPlotConfig({ locale: "pt-BR" });
  // Links diretos: #mercado e #evento abrem as outras abas; #e=<skin>@<AAAA-MM-DD>&limiar=..&penalidade=..&contaminacao=..
  const h = new URLSearchParams(location.hash.slice(1));
  if (h.has("limiar")) S.limiar = Math.min(6, Math.max(3, Math.round(+h.get("limiar") * 10) / 10));
  if (h.has("penalidade")) S.penalidade = Math.min(20, Math.max(5, Math.round(+h.get("penalidade"))));
  if (h.has("contaminacao")) S.contaminacao = Math.min(0.05, Math.max(0.01, Math.round(+h.get("contaminacao") * 200) / 200));
  if (h.has("amplitude")) S.amplitude = Math.min(30, Math.max(5, Math.round(+h.get("amplitude"))));
  if (h.has("volz")) S.volz = Math.min(4, Math.max(2, Math.round(+h.get("volz") * 2) / 2));
  if (h.has("reversao")) S.reversao = Math.min(14, Math.max(3, Math.round(+h.get("reversao"))));
  if (h.has("e")) {
    const [sk, dt] = h.get("e").split("@");
    S.skin = sk; S.sel = `${sk}@${dt}`; S.range = [noon(addDays(dt, -75)), noon(addDays(dt, 75))];
  }
  syncSliders(); renderMode(); bind(); carregar();
  if (h.has("evento")) document.querySelector('.tab[data-tab="evento"]').click();
  if (h.has("mercado")) document.querySelector('.tab[data-tab="mercado"]').click();
});
