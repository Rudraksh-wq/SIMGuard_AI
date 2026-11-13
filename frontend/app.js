"use strict";

const FIELDS = [
  "sim_swap_time_gap_minutes", "device_change_flag", "sim_type_change_flag", "imsi_change_flag",
  "iccid_change_flag", "otp_and_sim_change_geo_hash_length", "recent_sim_activation_days",
  "num_sim_changes_last_30d", "previous_sim_holder_tenure_days", "account_age_days", "ip_change_flag",
];
const LABELS = {
  sim_swap_time_gap_minutes: "OTP → SIM change gap (min)",
  device_change_flag: "Device changed",
  sim_type_change_flag: "SIM type changed",
  imsi_change_flag: "IMSI changed",
  iccid_change_flag: "ICCID changed",
  otp_and_sim_change_geo_hash_length: "Geohash similarity",
  recent_sim_activation_days: "Days since SIM activation",
  num_sim_changes_last_30d: "SIM changes, last 30 d",
  previous_sim_holder_tenure_days: "Previous holder tenure (d)",
  account_age_days: "Account age (d)",
  ip_change_flag: "IP changed",
};
const SAMPLES = {
  fraud: "SIM changed 45 minutes after an OTP request. Device changed, IMSI changed, ICCID changed, SIM type not changed. Geohash similarity 0.12. SIM activated 3 days ago, 2 SIM changes in last 30 days. Previous holder tenure 40 days, account age 120 days. IP changed.",
  normal: "time gap: 9500 minutes\ndevice changed: no\nsim type changed: no\nimsi changed: no\niccid changed: yes\ngeohash similarity: 0.94\ndays since sim activation: 400\nsim changes last 30d: 0\nprevious holder tenure: 900 days\naccount age: 1500 days\nip changed: no",
  json: JSON.stringify({ events: [
    { event_id: "evt-101", msisdn_hash: "a1f9…", sim_swap_time_gap_minutes: 30, device_change_flag: 1, sim_type_change_flag: 1, imsi_change_flag: 1, iccid_change_flag: 1, otp_and_sim_change_geo_hash_length: 0.05, recent_sim_activation_days: 2, num_sim_changes_last_30d: 3, previous_sim_holder_tenure_days: 20, account_age_days: 60, ip_change_flag: 1 },
    { event_id: "evt-102", msisdn_hash: "7c02…", sim_swap_time_gap_minutes: 8200, device_change_flag: 0, sim_type_change_flag: 0, imsi_change_flag: 0, iccid_change_flag: 1, otp_and_sim_change_geo_hash_length: 0.91, recent_sim_activation_days: 320, num_sim_changes_last_30d: 0, previous_sim_holder_tenure_days: 700, account_age_days: 1800, ip_change_flag: 0 },
  ] }, null, 2),
  csv: FIELDS.join(",") + ",sim_swap_flag\n" + [
    "9932,1,1,0,1,0.41,237,0,917,1623,1,1",
    "2577,0,0,0,1,0.02,271,0,595,75,0,0",
    "120,1,0,1,1,0.10,4,2,35,90,1,1",
    "7800,0,0,0,0,0.97,500,0,1200,2400,0,0",
    "4300,0,1,1,0,0.55,40,1,300,640,0,0",
  ].join("\n"),
};

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (v, d = 1) => (v === null || v === undefined ? "—" : Number(v).toFixed(d));
const pct = (v) => (v === null || v === undefined ? "—" : (v * 100).toFixed(1) + "%");
const badge = (v, text) => `<span class="badge b-${esc(v)}">${esc(text ?? String(v).replace(/_/g, " "))}</span>`;
const tierColor = (t) => ({ high: "var(--high)", medium: "var(--med)", low: "var(--low)" }[t] || "var(--accent)");

let mode = "text";
let file = null;
let health = null;
let batch = [];
let cases = [];
let caseFilter = "all";

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  let body = null;
  try { body = await res.json(); } catch { /* empty */ }
  if (!res.ok) {
    const d = body && body.detail;
    let msg = typeof d === "string" ? d : d && d.message ? d.message : `HTTP ${res.status}`;
    if (d && d.errors) msg += "\n" + d.errors.map((e) => `row ${e.row}: ${e.error}`).join("\n");
    if (d && d.expected_fields) msg += "\n\nExpected fields: " + d.expected_fields.join(", ");
    if (Array.isArray(d)) msg = d.map((e) => `${(e.loc || []).join(".")}: ${e.msg}`).join("\n");
    throw new Error(msg);
  }
  return body;
}

/* ---------- status ---------- */
async function loadHealth() {
  try {
    health = await api("/health");
    const set = (k, ok, title, text) => {
      const p = $(`.pill[data-k="${k}"]`);
      p.className = "pill " + (ok === null ? "" : ok ? "ok" : "bad");
      p.title = title;
      if (text) p.textContent = text;
    };
    if (health.mock_models) {
      ["ml", "laya", "qwen"].forEach((k) => set(k, null, "MOCK_MODELS=1"));
      set("ml", null, "MOCK_MODELS=1", "Mock models");
      return;
    }
    set("ml", health.ml.ok, health.ml.ok ? `LightGBM ${health.ml.lgbm_trees} trees · Isolation Forest ${health.ml.iforest_trees} trees` : health.ml.error,
      "LightGBM + IForest");
    const l = health.laya;
    set("laya", l.error ? false : l.loaded ? true : null, l.error || `${l.checkpoint} · gate ${fmt(l.gate.lo, 2)}/${fmt(l.gate.hi, 2)}`,
      l.loaded ? "Laya ready" : l.error ? "Laya error" : "Laya (loads on first use)");
    const q = health.qwen;
    set("qwen", q.reachable, `${q.model} · ${q.endpoint || "no link"} · ${q.detail} (click to change)`, q.reachable ? "Qwen 3.5 2B online" : q.configured ? "Qwen offline · fix link" : "Qwen · add link");
  } catch (e) {
    $$(".pill[data-k]").forEach((p) => { p.className = "pill bad"; p.title = "API unreachable"; });
  }
}

/* ---------- input ---------- */
function setMode(m) {
  mode = m;
  $$("#input-mode button").forEach((b) => b.classList.toggle("active", b.dataset.mode === m));
  $$(".mode").forEach((el) => el.classList.toggle("hidden", el.dataset.mode !== m));
}

function setFile(f) {
  file = f;
  $("#file-name").textContent = f ? `${f.name} · ${(f.size / 1024).toFixed(1)} KB` : "or click to choose · up to 5 MB, 2000 rows";
}

function showError(msg) {
  const el = $("#input-error");
  el.textContent = msg || "";
  el.classList.toggle("hidden", !msg);
}

async function run() {
  showError("");
  const btn = $("#run");
  let req;
  if (mode === "text") {
    const content = $("#prompt").value.trim();
    if (!content) return showError("Write a prompt or pick a sample.");
    req = api("/analyze/input", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ content, kind: "text" }) });
  } else if (mode === "paste") {
    const content = $("#paste").value.trim();
    if (!content) return showError("Paste CSV or JSON first.");
    req = api("/analyze/input", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ content }) });
  } else {
    if (!file) return showError("Choose a file first.");
    const fd = new FormData();
    fd.append("file", file);
    req = api("/analyze/upload", { method: "POST", body: fd });
  }
  btn.disabled = true;
  btn.innerHTML = '<span class="loading"><span class="spinner"></span>Analyzing…</span>';
  $("#detail").innerHTML = "";
  try {
    const out = await req;
    renderBatch(out);
    if (out.results.length) openCase(out.results[0], $("#detail"));
    pollPending(out.results);
  } catch (e) {
    showError(e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "Analyze";
    loadHealth();
  }
}

/* ---------- results table ---------- */
function rowHtml(r, i, withStatus) {
  return `<tr data-i="${i}">
    <td>${esc(r.event.event_id || r.case_id.slice(0, 8))}</td>
    <td>${badge(r.verdict)}</td>
    <td class="num"><b style="color:${tierColor(r.final.tier)}">${fmt(r.final.score)}</b></td>
    <td>${badge(r.final.tier)}</td>
    <td class="num">${pct(r.ml.lgbm_prob)}</td>
    <td class="num">${fmt(r.ml.anomaly, 2)}</td>
    <td class="num">${r.laya.p_fraud === null ? "—" : fmt(r.laya.p_fraud, 3)}</td>
    <td>${r.laya.gate ? badge(r.laya.gate) : badge("none", "n/a")}</td>
    <td>${badge(r.agreement)}</td>
    <td>${r.report_source === "none" ? "—" : r.report_source === "pending" ? badge("qwen", "Qwen writing…") : badge(r.report_source)}</td>
    ${withStatus ? `<td>${badge(r.status)}</td><td>${esc((r.created_at || "").replace("T", " ").slice(0, 19))}</td>` : ""}
  </tr>`;
}
const HEAD = `<th>Event</th><th>Verdict</th><th class="num">Risk</th><th>Tier</th><th class="num">LightGBM</th><th class="num">Anomaly</th><th class="num">Laya p</th><th>Laya gate</th><th>Agreement</th><th>Report</th>`;

function renderBatch(out) {
  batch = out.results;
  const flagged = batch.filter((r) => r.fraud_detected).length;
  $("#batch").classList.remove("hidden");
  $("#batch-summary").innerHTML =
    badge("none", `${out.source.toUpperCase()} input`) +
    badge("none", `${out.rows_analysed}/${out.rows_received} rows analysed`) +
    badge("flagged", `${flagged} flagged`) +
    badge("not_flagged", `${batch.length - flagged} not flagged`) +
    (out.errors.length ? badge("medium", `${out.errors.length} rows rejected`) : "");
  const notes = [...out.notes];
  if (out.errors.length) notes.push("Rejected rows: " + out.errors.slice(0, 5).map((e) => `row ${e.row}: ${e.error}`).join(" · ") + (out.errors.length > 5 ? " …" : ""));
  $("#batch-notes").innerHTML = notes.map((n) => `<p class="note">${esc(n)}</p>`).join("");
  const t = $("#batch-table");
  t.innerHTML = `<thead><tr>${HEAD}</tr></thead><tbody>${batch.map((r, i) => rowHtml(r, i, false)).join("")}</tbody>`;
  t.querySelector("tbody tr")?.classList.add("sel");
  t.onclick = (ev) => {
    const tr = ev.target.closest("tr[data-i]");
    if (!tr) return;
    $$("tr", t).forEach((x) => x.classList.toggle("sel", x === tr));
    openCase(batch[+tr.dataset.i], $("#detail"));
  };
}

/* ---------- report rendering ---------- */
function renderReport(text) {
  const lines = String(text || "").replace(/\r/g, "").split("\n");
  let html = "";
  let list = false;
  const close = () => { if (list) { html += "</ul>"; list = false; } };
  const inline = (s) => esc(s).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>");
  for (let raw of lines) {
    const line = raw.trim();
    if (!line) { close(); continue; }
    const heading = line.match(/^(?:#+\s*)?\**\s*(Summary|Supplied evidence|Evidence|Limitations|Suggested review steps|Recommended actions|Why it was flagged)\s*\**\s*:?\**\s*(.*)$/i);
    if (heading) {
      close();
      html += `<h4>${esc(heading[1])}</h4>`;
      if (heading[2]) html += `<p>${inline(heading[2].replace(/^\*+\s*/, ""))}</p>`;
      continue;
    }
    const bullet = line.match(/^(?:[-*•]|\d+[.)])\s+(.*)$/);
    if (bullet) {
      if (!list) { html += "<ul>"; list = true; }
      html += `<li>${inline(bullet[1])}</li>`;
      continue;
    }
    if (/^#+\s/.test(line)) { close(); html += `<h4>${esc(line.replace(/^#+\s*/, ""))}</h4>`; continue; }
    close();
    html += `<p>${inline(line)}</p>`;
  }
  close();
  return html;
}

/* ---------- case detail ---------- */
function meter(value, max, color, marker) {
  const w = Math.max(0, Math.min(100, (value / max) * 100));
  const m = marker === undefined ? "" : `<b style="left:${(marker / max) * 100}%"></b>`;
  return `<div class="meter"><i style="width:${w}%;background:${color}"></i>${m}</div>`;
}

function contributionBars(contrib, event) {
  const items = Object.entries(contrib || {});
  if (!items.length) return '<p class="legend">No contributions (mock mode).</p>';
  const max = Math.max(...items.map(([, v]) => Math.abs(v)), 0.01);
  return `<div class="bars">${items.map(([k, v]) => {
    const w = (Math.abs(v) / max) * 50;
    return `<div class="bar"><span class="name" title="${esc(k)}">${esc(LABELS[k] || k)} = <b>${esc(k.endsWith("_flag") ? (event[k] ? "yes" : "no") : event[k])}</b></span>
      <span class="track"><span class="fill ${v >= 0 ? "up" : "down"}" style="width:${w}%"></span></span>
      <span class="v">${v >= 0 ? "+" : ""}${fmt(v, 2)}</span></div>`;
  }).join("")}</div>
  <p class="legend">LightGBM per-feature contribution (log-odds). Red pushes toward fraud, green away from it.</p>`;
}

function featureTable(r) {
  const ctx = r.input.extra || {};
  const rows = FIELDS.map((f) => `<tr><td>${esc(f)}</td><td>${esc(r.event[f])}</td><td>${esc(LABELS[f])}</td></tr>`).join("");
  const ids = ["event_id", "msisdn_hash", "timestamp"].filter((k) => r.event[k]).map((k) => `<tr><td>${k}</td><td colspan="2">${esc(r.event[k])}</td></tr>`).join("");
  const extra = Object.entries(ctx).map(([k, v]) => `<tr><td>${esc(k)}</td><td colspan="2">${esc(v)} <span class="legend">(context only)</span></td></tr>`).join("");
  return `<table class="kv"><tbody>${ids}${rows}${extra}</tbody></table>`;
}

function reportBlock(r) {
  if (!r.report) {
    return `<p class="legend">No report: the case was not flagged and Laya did not ask for human review. Qwen only writes reports when needed, which keeps the pipeline fast.</p>
      <button class="ghost" data-act="report">Ask Qwen for a report anyway</button>`;
  }
  const meta = [
    badge(r.report_source, r.report_source === "qwen" ? `Qwen · ${r.report_model || ""}` : "Template report"),
    r.report_latency_ms != null ? `<span>${fmt(r.report_latency_ms / 1000, 1)} s</span>` : "",
    `<button class="link" data-act="report">${r.report_source === "qwen" ? "Regenerate" : "Try Qwen again"}</button>`,
  ].join("");
  if (r.report_source === "pending") {
    return `<div class="loading" style="margin-bottom:10px"><span class="spinner"></span>Qwen 3.5 2B is writing the report (about 15–20 s). The template report is shown until then.</div>
      <div class="report">${renderReport(r.report)}</div>`;
  }
  const err = r.report_error ? `<p class="note">${esc(r.report_error)}</p>` : "";
  return `${err}<div class="report">${renderReport(r.report)}</div><div class="report-meta" style="margin-top:10px">${meta}</div>`;
}

function openCase(r, host) {
  host.dataset.caseId = r.case_id;
  const laya = r.laya;
  const thr = health ? health.laya_fraud_threshold : 0.5;
  host.innerHTML = `
  <div class="verdict ${esc(r.verdict)}">
    <div>
      <div class="big">${r.verdict === "flagged" ? "Flagged: suspected SIM-swap fraud" : r.verdict === "not_flagged" ? "Not flagged" : "Inconclusive"}</div>
      <div class="sub">Case ${esc(r.case_id.slice(0, 12))} · ${esc(r.input.source)} input · models ${badge(r.agreement)} · status ${badge(r.status)}</div>
    </div>
    <div class="review">
      <button class="btn-confirm" data-act="confirmed">Confirm fraud</button>
      <button class="btn-reject" data-act="false_alarm">False alarm</button>
    </div>
  </div>

  <div class="scores">
    <div class="score"><div class="label">Final risk score</div>
      <div class="value" style="color:${tierColor(r.final.tier)}">${fmt(r.final.score)}</div>
      <div class="meta">${badge(r.final.tier)} · Laya weight ${fmt(r.final.laya_weight_used, 2)}</div>
      ${meter(r.final.score, 100, tierColor(r.final.tier), 70)}</div>
    <div class="score"><div class="label">LightGBM fraud probability</div>
      <div class="value">${pct(r.ml.lgbm_prob)}</div>
      <div class="meta">ML risk score ${fmt(r.ml.risk_score)} (70% LightGBM + 30% anomaly)</div>
      ${meter(r.ml.lgbm_prob, 1, "var(--accent)")}</div>
    <div class="score"><div class="label">Isolation Forest anomaly</div>
      <div class="value">${fmt(r.ml.anomaly, 2)}</div>
      <div class="meta">0 = typical of normal traffic · 1 = very unusual</div>
      ${meter(r.ml.anomaly, 1, "var(--med)")}</div>
    <div class="score"><div class="label">Laya p(fraud)</div>
      <div class="value">${laya.p_fraud === null ? "—" : fmt(laya.p_fraud, 3)}</div>
      <div class="meta">${laya.gate ? "gate " + badge(laya.gate) : esc(laya.error || "unavailable")}</div>
      ${laya.p_fraud === null ? "" : meter(laya.p_fraud, 1, "var(--high)", thr)}</div>
  </div>

  <div class="two">
    <div class="card"><h3>Analyst report</h3><div id="report-slot">${reportBlock(r)}</div></div>
    <div>
      <div class="card"><h3>Why the score is what it is</h3>${contributionBars(r.ml.contributions, r.event)}</div>
      <div class="card"><h3>Evidence indicators</h3>
        ${r.evidence.length ? `<ul class="evidence">${r.evidence.map((e) => `<li>${esc(e)}</li>`).join("")}</ul>` : '<p class="legend">No rule-based indicators fired.</p>'}
      </div>
    </div>
  </div>

  <div class="two">
    <div class="card"><h3>Input → parsed features</h3>
      ${(r.input.notes || []).length ? `<details class="box"><summary>Parser notes (${r.input.notes.length})</summary><ul class="evidence">${r.input.notes.map((n) => `<li>${esc(n)}</li>`).join("")}</ul></details>` : ""}
      ${featureTable(r)}
      ${r.input.raw ? `<details class="box"><summary>Original input</summary><pre class="json">${esc(r.input.raw)}</pre></details>` : ""}
    </div>
    <div class="card"><h3>What Qwen received</h3>
      ${r.report_payload ? `<pre class="json">${esc(JSON.stringify(r.report_payload, null, 2))}</pre>` : '<p class="legend">Nothing was sent to Qwen for this case.</p>'}
      <h3 style="margin-top:14px">Latency</h3>
      <div class="latency">
        <span>ML <b>${fmt(r.ml.latency_ms, 2)} ms</b></span>
        <span>Laya <b>${fmt(laya.latency_ms, 1)} ms</b></span>
        <span>Qwen <b>${r.report_latency_ms == null ? "—" : fmt(r.report_latency_ms / 1000, 2) + " s"}</b></span>
        <span>model <b>${esc(laya.model)}</b></span>
      </div>
    </div>
  </div>`;

  host.onclick = async (ev) => {
    const act = ev.target.closest("[data-act]")?.dataset.act;
    if (!act) return;
    try {
      let updated;
      if (act === "report") {
        $("#report-slot", host).innerHTML = '<div class="loading"><span class="spinner"></span>Qwen 3.5 2B is writing the report…</div>';
        updated = await api(`/cases/${r.case_id}/report`, { method: "POST" });
      } else {
        updated = await api(`/cases/${r.case_id}/review`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ status: act }) });
        updated = { ...r, ...updated };
      }
      replaceCase(updated);
      openCase(updated, host);
    } catch (e) {
      alert(e.message);
      openCase(r, host);
    }
  };
  if (!host.dataset.quiet) host.scrollIntoView({ behavior: "smooth", block: "start" });
}

function openCaseQuiet(r, host) {
  host.dataset.quiet = "1";
  openCase(r, host);
  delete host.dataset.quiet;
}

function pollPending(list) {
  const waiting = new Set(list.filter((r) => r.report_source === "pending").map((r) => r.case_id));
  if (!waiting.size) return;
  const started = Date.now();
  const tick = async () => {
    for (const id of [...waiting]) {
      try {
        const c = await api(`/cases/${id}`);
        if (c.report_source !== "pending") { waiting.delete(id); replaceCase(c, true); }
      } catch { waiting.delete(id); }
    }
    if (waiting.size && Date.now() - started < 5 * 60 * 1000) setTimeout(tick, 3000);
  };
  setTimeout(tick, 3000);
}

function replaceCase(u, refreshOpen) {
  if (refreshOpen) {
    ["#detail", "#case-detail"].forEach((sel) => { const h = $(sel); if (h.dataset.caseId === u.case_id) openCaseQuiet(u, h); });
  }
  const i = batch.findIndex((x) => x.case_id === u.case_id);
  if (i >= 0) {
    batch[i] = u;
    const tr = $(`#batch-table tr[data-i="${i}"]`);
    if (tr) { tr.outerHTML = rowHtml(u, i, false); $(`#batch-table tr[data-i="${i}"]`).classList.add("sel"); }
  }
  const j = cases.findIndex((x) => x.case_id === u.case_id);
  if (j >= 0) { cases[j] = u; renderCases(); }
}

/* ---------- case queue ---------- */
async function loadCases() {
  try { cases = await api("/cases"); } catch (e) { cases = []; }
  renderCases();
  pollPending(cases);
}
function renderCases() {
  const list = cases.filter((c) =>
    caseFilter === "all" ? true :
    caseFilter === "flagged" ? c.fraud_detected :
    caseFilter === "pending" ? c.status === "pending" && c.fraud_detected :
    c.status !== "pending");
  const t = $("#case-table");
  if (!list.length) { t.innerHTML = '<tbody><tr><td class="legend">No cases yet. Analyze something first.</td></tr></tbody>'; return; }
  t.innerHTML = `<thead><tr>${HEAD}<th>Status</th><th>Created</th></tr></thead><tbody>${list.map((c) => rowHtml(c, cases.indexOf(c), true)).join("")}</tbody>`;
  t.onclick = (ev) => {
    const tr = ev.target.closest("tr[data-i]");
    if (!tr) return;
    $$("tr", t).forEach((x) => x.classList.toggle("sel", x === tr));
    openCase(cases[+tr.dataset.i], $("#case-detail"));
  };
}

/* ---------- Qwen connection ---------- */
function qwenInfo(st) {
  const row = (k, v) => `<tr><td>${k}</td><td>${v ? esc(v) : '<span class="legend">—</span>'}</td></tr>`;
  const c = st.check || {};
  return `<p>${c.reachable ? badge("not_flagged", "online") : badge("flagged", "offline")} ${esc(c.detail || "")}</p>
    <table class="kv"><tbody>
      ${row("In use", st.active || c.endpoint)}
      ${row("Pasted link", st.override)}
      ${row("Discovery topic", st.discovery)}
      ${row("Discovered link", st.discovered)}
      ${row(".env QWEN_ENDPOINT", st.env_endpoint)}
      ${row("User", st.user + (st.password_set ? " · password set" : " · NO password"))}
    </tbody></table>
    <p class="legend">Links are tried in that order; a dead link is skipped and the next one that answers is used.</p>`;
}

async function openQwenDialog() {
  const dlg = $("#qwen-dialog");
  $("#qwen-msg").className = "hidden";
  $("#qwen-info").innerHTML = '<div class="loading"><span class="spinner"></span>Checking Qwen…</div>';
  dlg.showModal();
  try { $("#qwen-info").innerHTML = qwenInfo(await api("/settings/qwen")); }
  catch (e) { $("#qwen-info").textContent = e.message; }
}

async function saveQwen(clear) {
  const msg = $("#qwen-msg");
  const btn = $("#qwen-save");
  btn.disabled = true;
  msg.className = "loading"; msg.innerHTML = '<span class="spinner"></span>Testing the link…';
  try {
    const st = clear
      ? await api("/settings/qwen", { method: "DELETE" })
      : await api("/settings/qwen", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ endpoint: $("#qwen-url").value.trim() }) });
    $("#qwen-info").innerHTML = qwenInfo(st);
    msg.className = "ok-msg"; msg.textContent = clear ? "Pasted link removed." : "Saved. New reports use this link.";
    $("#qwen-url").value = "";
    loadHealth();
  } catch (e) {
    msg.className = "error"; msg.textContent = e.message;
  } finally { btn.disabled = false; }
}

/* ---------- wiring ---------- */
function init() {
  $("#field-list").innerHTML = FIELDS.map((f) => `<code title="${esc(LABELS[f])}">${f}</code>`).join("");
  $$("#input-mode button").forEach((b) => (b.onclick = () => setMode(b.dataset.mode)));
  $$("[data-sample]").forEach((b) => (b.onclick = () => {
    const s = b.dataset.sample;
    if (s === "fraud" || s === "normal") { setMode("text"); $("#prompt").value = SAMPLES[s]; }
    else { setMode("paste"); $("#paste").value = SAMPLES[s]; }
  }));
  $("#run").onclick = run;
  $('.pill[data-k="qwen"]').onclick = openQwenDialog;
  $("#qwen-save").onclick = () => saveQwen(false);
  $("#qwen-clear").onclick = () => saveQwen(true);
  $("#prompt").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) run(); });
  $("#file").onchange = (e) => setFile(e.target.files[0] || null);
  const drop = $("#drop");
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0] || null));
  $$(".tab").forEach((b) => (b.onclick = () => {
    $$(".tab").forEach((x) => x.classList.toggle("active", x === b));
    $$(".view").forEach((v) => v.classList.toggle("hidden", v.id !== "view-" + b.dataset.view));
    if (b.dataset.view === "cases") loadCases();
  }));
  $$("#case-filter button").forEach((b) => (b.onclick = () => {
    caseFilter = b.dataset.f;
    $$("#case-filter button").forEach((x) => x.classList.toggle("active", x === b));
    renderCases();
  }));
  loadHealth();
  setInterval(loadHealth, 30000);
  // demo deep link: /#sample=fraud|normal|json|csv fills the input and runs it
  const sample = new URLSearchParams(location.hash.slice(1)).get("sample");
  if (SAMPLES[sample]) { $(`[data-sample="${sample}"]`).click(); run(); }
  if (location.hash === "#qwen") openQwenDialog();
}
init();
