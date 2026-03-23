/* ═══════════════════════════════════════════════════════
   BEAST v2.0 — Clinical Decision Support System
   app.js  ·  Full Application Logic  ·  v3
   ═══════════════════════════════════════════════════════ */
'use strict';

const API_URL = 'http://localhost:8000/predict';
const VITALS = [
  { id:'gluc', name:'Glucose',      unit:'mg/dL',  min:40,  max:600, lo:70,   hi:140,  def:105,  step:1   },
  { id:'crea', name:'Creatinine',   unit:'mg/dL',  min:0.1, max:15,  lo:0.6,  hi:1.2,  def:0.9,  step:0.1 },
  { id:'hemo', name:'Hemoglobin',   unit:'g/dL',   min:5,   max:20,  lo:13.5, hi:17.5, def:14.0, step:0.1 },
  { id:'wbc',  name:'WBC Count',    unit:'K/uL',   min:1,   max:50,  lo:4.5,  hi:11.0, def:7.0,  step:0.1 },
  { id:'hr',   name:'Heart Rate',   unit:'bpm',    min:30,  max:200, lo:60,   hi:100,  def:78,   step:1   },
  { id:'sbp',  name:'Systolic BP',  unit:'mmHg',   min:70,  max:220, lo:90,   hi:140,  def:120,  step:1   },
  { id:'spo2', name:'SpO₂',         unit:'%',      min:70,  max:100, lo:94,   hi:100,  def:97.0, step:0.1 },
  { id:'temp', name:'Temperature',  unit:'°C',     min:34,  max:42,  lo:36.1, hi:37.2, def:36.6, step:0.1 },
];

const predictionHistory = [];

/* ═══════ INIT ════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {
  buildVitalsGrid();
  setDefaultDate();
  syncBreadcrumb();
  buildLoadingOverlay();
  ['pid','age','gender'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('input', syncBreadcrumb);
  });
  // Make sure inputs tab is visible on load
  showTab('inputs');
});

function setDefaultDate() {
  const el = document.getElementById('admit-date');
  if (el) el.value = new Date().toISOString().split('T')[0];
}

function syncBreadcrumb() {
  const pid = document.getElementById('pid')?.value || '—';
  const age = document.getElementById('age')?.value || '—';
  const bc  = document.getElementById('breadcrumb');
  if (bc) bc.innerHTML =
    `<span class="bc-item">${pid}</span><span class="bc-sep">›</span>` +
    `<span class="bc-item">Age ${age}</span><span class="bc-sep">›</span>` +
    `<span class="bc-item active">Risk Assessment</span>`;
}

function buildLoadingOverlay() {
  if (document.getElementById('loading-overlay')) return;
  const div = document.createElement('div');
  div.className = 'loading-overlay';
  div.id = 'loading-overlay';
  div.innerHTML = `
    <div class="loading-box">
      <div class="loading-hex"></div>
      <div class="loading-title">Running Inference…</div>
      <div class="loading-sub" id="loading-sub">Preprocessing patient data</div>
    </div>`;
  document.body.appendChild(div);
}

/* ═══════ TAB SWITCHING ═══════════════════════════════ */
function showTab(name) {
  // Hide all content panels
  document.querySelectorAll('.tab-content').forEach(el => {
    el.style.display = 'none';
  });
  // Deactivate all tab buttons
  document.querySelectorAll('.tab').forEach(el => {
    el.classList.remove('active');
  });
  // Show selected panel
  const panel = document.getElementById('tab-' + name);
  if (panel) panel.style.display = 'flex';
  // Activate selected button
  const btn = document.querySelector('.tab[data-tab="' + name + '"]');
  if (btn) btn.classList.add('active');
}

/* ═══════ VITALS GRID ══════════════════════════════════ */
function buildVitalsGrid() {
  const grid = document.getElementById('vitals-grid');
  if (!grid) return;
  grid.innerHTML = '';

  VITALS.forEach(v => {
    const card = document.createElement('div');
    card.className = 'vital-card';
    card.id = 'vcard-' + v.id;
    card.innerHTML = `
      <div class="vital-top">
        <span class="vital-name">${v.name}</span>
        <span class="vital-flag-dot" id="vdot-${v.id}"></span>
      </div>
      <div class="vital-value-row">
        <span class="vital-val" id="vval-${v.id}">${fmtV(v.def, v.step)}</span>
        <span class="vital-unit">${v.unit}</span>
      </div>
      <div class="vital-range">Normal: ${v.lo}–${v.hi} ${v.unit}</div>
      <input type="range" class="vital-slider" id="vs-${v.id}"
        min="${v.min}" max="${v.max}" value="${v.def}" step="${v.step}" />
      <div class="vital-status" id="vstatus-${v.id}"></div>`;
    grid.appendChild(card);

    const slider = document.getElementById('vs-' + v.id);
    slider.addEventListener('input', () => refreshVital(v.id));
    refreshVital(v.id); // paint initial state
  });
}

function fmtV(val, step) {
  return step < 1 ? parseFloat(val).toFixed(1) : String(Math.round(parseFloat(val)));
}

function refreshVital(id) {
  const v       = VITALS.find(x => x.id === id);
  const slider  = document.getElementById('vs-' + id);
  const valEl   = document.getElementById('vval-' + id);
  const dotEl   = document.getElementById('vdot-' + id);
  const statEl  = document.getElementById('vstatus-' + id);
  const cardEl  = document.getElementById('vcard-' + id);
  if (!v || !slider) return;

  const raw     = parseFloat(slider.value);
  const inRange = raw >= v.lo && raw <= v.hi;
  const below   = raw < v.lo;

  const C_OK   = '#10b981';
  const C_WARN = '#f59e0b';
  const C_BAD  = '#f43f5e';
  const col    = inRange ? C_OK : (below ? C_BAD : C_WARN);

  // Update displayed value
  valEl.textContent  = fmtV(raw, v.step);
  valEl.style.color  = col;

  // Update dot
  dotEl.style.background = col;
  dotEl.style.boxShadow  = `0 0 6px ${col}`;

  // Update status text
  statEl.textContent = inRange ? '✓ Within range' : (below ? '↓ Below normal' : '↑ Above normal');
  statEl.style.color = col;

  // Update card border
  cardEl.style.borderColor = inRange ? 'var(--border)' : col + '55';

  // Update slider accent colour via CSS filter trick
  slider.style.accentColor = col;

  refreshVitalsBadge();
}

function refreshVitalsBadge() {
  let ok = 0;
  VITALS.forEach(v => {
    const val = parseFloat(document.getElementById('vs-' + v.id)?.value ?? v.def);
    if (val >= v.lo && val <= v.hi) ok++;
  });
  const badge = document.getElementById('vitals-badge');
  if (!badge) return;
  badge.textContent = ok + ' / ' + VITALS.length + ' within range';
  badge.className = 'section-badge' + (ok < VITALS.length ? ' warning' : '');
}

function getVitalsValues() {
  return VITALS.map(v => parseFloat(document.getElementById('vs-' + v.id)?.value ?? v.def));
}

/* ═══════ API CALL ════════════════════════════════════ */
async function callPredictAPI(payload) {
  try {
    const res = await fetch(API_URL, {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(8000),
    });
    if (!res.ok) {
      const txt = await res.text().catch(()=>'');
      return { success:false, error:`HTTP ${res.status}: ${txt.slice(0,200)}` };
    }
    const data = await res.json();
    let score = data.risk_score ?? data.probability ?? data.score ?? null;
    if (score === null) return { success:false, error:"API missing 'risk_score' field." };
    if (score <= 1.0) score = Math.round(score*1000)/10;
    return { success:true, risk_score:score, feature_importance: data.feature_importance || null };
  } catch(err) {
    if (err.name==='TimeoutError'||err.name==='AbortError')
      return { success:false, error:'Request timed out. Check server load.' };
    return { success:false, error:`Cannot reach ${API_URL}. Is FastAPI running?` };
  }
}

/* ═══════ RUN PREDICTION ══════════════════════════════ */
async function runPrediction() {
  const vitals = getVitalsValues();
  const payload = {
    patient_id: document.getElementById('pid')?.value || '—',
    age:        document.getElementById('age')?.value || '—',
    gender:     document.getElementById('gender')?.value || '—',
    diagnosis:  document.getElementById('diagnosis')?.value || '—',
    medication: document.getElementById('medication')?.value || '—',
    vitals,
  };

  setLoading(true);
  await sleep(600); // ✅ FIXED: Removed DEMO_MODE reference here
  const result = await callPredictAPI(payload);
  setLoading(false);

  // ✅ THIS IS WHY IT DOESNT SWITCH TABS IF THE API FAILS:
  if (!result.success) {
    showToast('Prediction failed: ' + result.error, 'error');
    return; // Stops the code dead in its tracks!
  }

  let fi = result.feature_importance || VITALS.map((v,i)=>({
    name:v.name, unit:v.unit, value:vitals[i],
    weight:0.5, inRange:vitals[i]>=v.lo&&vitals[i]<=v.hi, step:v.step
  }));

  paintResults(result.risk_score, fi, payload);
  addHistory({ pid:payload.patient_id, diag:payload.diagnosis, score:result.risk_score });
  showTab('results'); // Switches the tab!
  showToast('Prediction complete — Risk: ' + result.risk_score.toFixed(1) + '%', 'success');
}

/* ── Payload ────────────────────────────────────────── */
function paintPayload(p, score, fi) {
  const s = document.getElementById('payload-sent');
  const r = document.getElementById('payload-received');
  if (s) s.textContent = JSON.stringify({
    patient_id:p.patient_id, age:p.age, gender:p.gender,
    diagnosis:p.diagnosis, medication:p.medication,
    vitals:p.vitals.map(v=>Math.round(v*10)/10)
  },null,2);
  if (r) r.textContent = JSON.stringify({
    risk_score: score,
    risk_pct:   score.toFixed(1)+'%',
    top_driver: fi[0]?.name || '—',
    feature_importance: fi.slice(0,4).map(f=>({
      feature:f.name, value:fmtV(f.value,f.step||1), unit:f.unit,
      weight: Math.round(f.weight*1000)/1000
    })),
    timestamp: new Date().toISOString(), // ✅ FIXED: Removed demo_mode line
  },null,2);
}

/* ═══════ RENDER RESULTS ══════════════════════════════ */
/* ═══════ RENDER RESULTS (Updated & Crash-Proof) ══════════════════════════════ */
function paintResults(score, fi, payload) {
  try {
    // 1. Grab the elements
    const ph = document.getElementById('results-placeholder');
    const rc = document.getElementById('results-content');
    
    // 2. Hide the placeholder
    if (ph) ph.style.display = 'none';
    
    // 3. AGGRESSIVELY unhide the results
    if (rc) { 
      rc.classList.remove('hidden'); // Strip the CSS class
      rc.className = 'results-content'; // Double-kill any rogue hidden classes
      rc.style.display = 'flex'; 
      rc.style.flexDirection = 'column'; 
      rc.style.gap = '20px'; 
    }

    // 4. Ensure data types are safe to prevent rendering crashes
    const safeScore = Number(score) || 0;
    const safeFi = Array.isArray(fi) ? fi : [];

    // 5. Paint the components
    drawGauge(safeScore);
    paintPriority(safeScore, payload);
    paintSnapshot(safeFi);
    paintReasoning(safeFi);
    paintPayload(payload, safeScore, safeFi);

  } catch (error) {
    console.error("UI Rendering crashed:", error);
    showToast("Error drawing results! Check the browser console.", "error");
  }
}

/* ── Gauge ──────────────────────────────────────────── */
function drawGauge(score) {
  const canvas = document.getElementById('gaugeCanvas');
  const scoreEl = document.getElementById('gauge-score');
  if (!canvas) return;

  const ctx = canvas.getContext('2d');
  const W=220, H=130, cx=110, cy=118, R=90;
  ctx.clearRect(0,0,W,H);

  // Zone arcs
  [[0,30,'rgba(16,185,129,0.12)'],[30,65,'rgba(245,158,11,0.12)'],[65,100,'rgba(244,63,94,0.12)']].forEach(([lo,hi,c])=>{
    ctx.beginPath();
    ctx.arc(cx,cy,R, Math.PI+(lo/100)*Math.PI, Math.PI+(hi/100)*Math.PI);
    ctx.lineWidth=20; ctx.strokeStyle=c; ctx.stroke();
  });

  // Track
  ctx.beginPath();
  ctx.arc(cx,cy,R, Math.PI, 2*Math.PI);
  ctx.lineWidth=10; ctx.strokeStyle='rgba(255,255,255,0.04)'; ctx.stroke();

  // Fill
  const col = score<30?'#10b981':score<65?'#f59e0b':'#f43f5e';
  ctx.beginPath();
  ctx.arc(cx,cy,R, Math.PI, Math.PI+(score/100)*Math.PI);
  ctx.lineWidth=10; ctx.lineCap='round'; ctx.strokeStyle=col; ctx.stroke();

  // End dot
  const ea = Math.PI+(score/100)*Math.PI;
  ctx.beginPath();
  ctx.arc(cx+R*Math.cos(ea), cy+R*Math.sin(ea), 5, 0, 2*Math.PI);
  ctx.fillStyle=col; ctx.fill();

  if (scoreEl) { scoreEl.textContent=score.toFixed(1)+'%'; scoreEl.style.color=col; }
}

/* ── Priority ───────────────────────────────────────── */
function paintPriority(score, p) {
  const bw = document.getElementById('badge-wrap');
  const pd = document.getElementById('priority-desc');
  const pm = document.getElementById('priority-meta');
  if (!bw) return;

  const [cls,icon,txt,desc] = score<30
    ? ['stable',  '●','Stable',        'Low readmission risk. Routine monitoring and standard discharge protocols are advised.']
    : score<65
    ? ['elevated','▲','Elevated Risk',  'Moderate readmission risk. Enhanced monitoring and care plan review are recommended.']
    : ['critical','⬥','Critical',       'High readmission risk. Immediate clinical review required. Consider specialist referral.'];

  bw.innerHTML = `<span class="priority-badge ${cls}">${icon} ${txt}</span>`;
  pd.textContent = desc;
  const ts = new Date().toLocaleTimeString('en-IN',{hour:'2-digit',minute:'2-digit'});
  pm.innerHTML =
    `<div class="pm-row"><span>Patient</span><span>${p.patient_id}</span></div>` +
    `<div class="pm-row"><span>Diagnosis</span><span>${p.diagnosis}</span></div>` +
    `<div class="pm-row"><span>Medication</span><span>${p.medication}</span></div>` +
    `<div class="pm-row"><span>Time</span><span>${ts}</span></div>`;
}

/* ── Snapshot Bars ──────────────────────────────────── */
function paintSnapshot(fi) {
  const c = document.getElementById('snapshot-bars');
  if (!c) return;
  c.innerHTML = fi.slice(0,8).map(item => {
    const vc = VITALS.find(v=>v.name===item.name);
    if (!vc) return '';
    const pct = Math.max(2,Math.min(97, ((item.value-vc.min)/(vc.max-vc.min))*100 ));
    const col = item.inRange ? '#10b981' : '#f43f5e';
    return `<div class="snap-row">
      <span class="snap-label">${item.name}</span>
      <div class="snap-track"><div class="snap-fill" style="width:${pct.toFixed(1)}%;background:${col};"></div></div>
      <span class="snap-val" style="color:${col};">${fmtV(item.value,item.step||1)} ${item.unit}</span>
    </div>`;
  }).join('');
}

/* ── Reasoning ──────────────────────────────────────── */
function paintReasoning(fi) {
  const grid = document.getElementById('reasoning-grid');
  const tb   = document.getElementById('top-driver-badge');
  if (!grid) return;

  const maxW = fi[0]?.weight || 1;
  const sevs = ['Primary driver','Major contributor','Major contributor','Moderate','Minor','Low','Low','Minimal'];
  grid.innerHTML = fi.slice(0,8).map((f,i) => {
    const pct = Math.round((f.weight/maxW)*100);
    const col = f.inRange ? '#10b981' : '#f43f5e';
    const cls = f.inRange ? 'ok' : (i===0?'danger':'warning');
    return `<div class="r-card ${cls}">
      <span class="r-rank-num">0${i+1}</span>
      <div class="r-body">
        <div class="r-name">${f.name}</div>
        <div class="r-sub">${sevs[i]||'—'} · <span style="color:${col}">${f.inRange?'✓ Normal':'⚠ Abnormal'}</span></div>
        <div class="r-bar-track"><div class="r-bar-fill" style="width:${pct}%;background:${col};"></div></div>
      </div>
      <div class="r-val-col">${fmtV(f.value,f.step||1)}<br><span style="font-size:9px;color:var(--text-muted);">${f.unit}</span></div>
    </div>`;
  }).join('');

  if (tb && fi[0]) {
    tb.textContent = 'Primary: ' + fi[0].name + ' (' + fmtV(fi[0].value,fi[0].step||1) + ' ' + fi[0].unit + ')';
    tb.className = 'section-badge' + (fi[0].inRange ? '' : ' warning');
  }
}

/* ═══════ HISTORY ═════════════════════════════════════ */
function addHistory({pid,diag,score}) {
  predictionHistory.unshift({pid,diag,score,ts:new Date()});
  const tbody = document.getElementById('history-tbody');
  if (!tbody) return;
  const emptyRow = tbody.querySelector('tr td[colspan]');
  if (emptyRow) emptyRow.closest('tr').remove();

  const col = score<30?'#10b981':score<65?'#f59e0b':'#f43f5e';
  const lbl = score<30?'Stable':score<65?'Elevated':'Critical';
  const ts  = predictionHistory[0].ts.toLocaleString('en-IN',{day:'2-digit',month:'short',hour:'2-digit',minute:'2-digit'});
  const tr  = document.createElement('tr');
  tr.innerHTML =
    `<td class="mono">${ts}</td><td class="mono">${pid}</td><td>${diag}</td>` +
    `<td class="mono" style="color:${col}">${score.toFixed(1)}%</td>` +
    `<td><span style="display:inline-block;padding:3px 10px;border-radius:99px;font-size:11px;font-weight:600;` +
    `background:${col}22;color:${col};border:1px solid ${col}55;">${lbl}</span></td>` +
    `<td><button class="history-view-btn" onclick="showTab('results')">View</button></td>`;
  tbody.insertBefore(tr, tbody.firstChild);
}

/* ═══════ UI HELPERS ══════════════════════════════════ */
function setLoading(on) {
  const el = document.getElementById('loading-overlay');
  if (!el) return;
  el.classList.toggle('active', on);
  if (on) {
    const msgs = ['Preprocessing patient data…','Running feature extraction…','Computing risk ensemble…','Generating clinical reasoning…'];
    let i=0;
    const sub = document.getElementById('loading-sub');
    const iv = setInterval(()=>{
      if (!el.classList.contains('active')){clearInterval(iv);return;}
      if (sub && i<msgs.length) sub.textContent = msgs[i++];
    },300);
  }
}

function showToast(msg, type='success') {
  const t = document.getElementById('toast');
  if (!t) return;
  t.textContent = msg;
  t.className = 'toast ' + type + ' show';
  setTimeout(()=>t.classList.remove('show'), 3500);
}

function sleep(ms){ return new Promise(r=>setTimeout(r,ms)); }
