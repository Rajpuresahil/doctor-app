const $ = s => document.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const escRe = s => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

const S = { chunks: [], pending: 0, summary: null, speech: [], capturedFrames: 0, sentFrames: 0, queue: Promise.resolve(), mic: null, hl: null };

/* ------------------------------------------------------------------ VAD
   Energy VAD with adaptive noise floor, hysteresis, pre-roll and hangover.
   20 ms frames @16 kHz. Only speech segments ever leave the browser. */
class VAD {
  constructor(onSegment, onFrame) {
    Object.assign(this, { onSegment, onFrame, noise: -60, pre: [], buf: [], speaking: false, run: 0, sil: 0 });
  }
  push(f) {
    let e = 0; for (let i = 0; i < f.length; i++) e += f[i] * f[i];
    const db = 20 * Math.log10(Math.sqrt(e / f.length) + 1e-9);
    const thr = Math.max(this.noise + 10, -50);
    const speech = db > thr;
    S.capturedFrames++;
    if (!this.speaking) {
      this.noise = Math.min(-35, Math.max(-75, this.noise * 0.95 + db * 0.05));
      this.pre.push(f); if (this.pre.length > 15) this.pre.shift();      // 300 ms pre-roll
      this.run = speech ? this.run + 1 : 0;
      if (this.run >= 4) { this.speaking = true; this.buf = [...this.pre]; this.sil = 0; }   // 80 ms to trigger
    } else {
      this.buf.push(f);
      this.sil = speech ? 0 : this.sil + 1;
      if (this.sil >= 35 || this.buf.length >= 600) this.finish();      // 700 ms hangover / 12 s cap
    }
    this.onFrame(db, thr, this.speaking);
  }
  finish() {
    if (this.buf.length - this.sil >= 15) {                              // >= 300 ms of speech
      const out = new Float32Array(this.buf.length * 320);
      this.buf.forEach((f, i) => out.set(f, i * 320));
      this.onSegment(out);
    }
    this.speaking = false; this.buf = []; this.pre = []; this.run = 0; this.sil = 0;
  }
}

/* ------------------------------------------------------------------ mic
   ScriptProcessor -> naive average downsample to 16 kHz -> VAD */
async function startMic() {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
  const ctx = new AudioContext();
  const src = ctx.createMediaStreamSource(stream);
  const proc = ctx.createScriptProcessor(4096, 1, 1);
  const ratio = ctx.sampleRate / 16000;
  let left = new Float32Array(0), frac = 0, fbuf = [];
  const vad = new VAD(onSegment, drawMeter);
  proc.onaudioprocess = e => {
    const inp = e.inputBuffer.getChannelData(0);
    const x = new Float32Array(left.length + inp.length); x.set(left); x.set(inp, left.length);
    let pos = frac;
    while (pos + ratio <= x.length) {
      let s = 0, n = 0;
      for (let j = Math.floor(pos); j < Math.floor(pos + ratio); j++) { s += x[j]; n++; }
      fbuf.push(s / (n || 1)); pos += ratio;
      if (fbuf.length === 320) { vad.push(Float32Array.from(fbuf)); fbuf = []; }
    }
    left = x.slice(Math.floor(pos)); frac = pos - Math.floor(pos);
  };
  src.connect(proc); proc.connect(ctx.destination);
  S.mic = { stream, ctx, proc, src, vad };
}
function stopMic() {
  const m = S.mic; if (!m) return;
  m.vad.finish(); m.proc.disconnect(); m.src.disconnect();
  m.stream.getTracks().forEach(t => t.stop()); m.ctx.close(); S.mic = null;
}

/* ------------------------------------------------------------------ meter */
const hist = [];
function drawMeter(db, thr, speaking) {
  hist.push([db, speaking]); if (hist.length > 150) hist.shift();
  const c = $("#meter"), g = c.getContext("2d"), W = c.width, H = c.height;
  g.clearRect(0, 0, W, H);
  const y = d => H - Math.max(0, Math.min(1, (d + 80) / 80)) * H;
  hist.forEach(([d, sp], i) => { g.fillStyle = sp ? "#16a34a" : "#b6c0d8"; g.fillRect(i * (W / 150), y(d), W / 150 - 1, H - y(d)); });
  g.strokeStyle = "#5b63f5"; g.setLineDash([4, 3]); g.beginPath(); g.moveTo(0, y(thr)); g.lineTo(W, y(thr)); g.stroke(); g.setLineDash([]);
  const v = $("#vadstate"); v.textContent = speaking ? "speech" : "silence"; v.className = speaking ? "on" : "";
  updateStats();
}

/* ------------------------------------------------------------------ segments -> Whisper (ordered) */
function onSegment(f32) {
  S.sentFrames += f32.length / 320;
  const pcm = new Int16Array(f32.length);
  for (let i = 0; i < f32.length; i++) pcm[i] = Math.max(-1, Math.min(1, f32[i])) * 32767;
  S.speech.push(pcm);
  const idx = S.chunks.length; S.chunks.push(""); S.pending++; renderTranscript();
  const ctxHdr = S.chunks.join(" ").slice(-250).replace(/[^\x20-\x7E]/g, " ");
  S.queue = S.queue.then(async () => {
    try {
      const r = await fetch("/api/transcribe", { method: "POST", headers: { "Content-Type": "application/octet-stream", "X-Context": ctxHdr }, body: pcm.buffer });
      const j = await r.json();
      if (!r.ok) throw new Error(j.error || r.status);
      S.chunks[idx] = j.text || "";
    } catch (e) { S.chunks[idx] = ""; toast("ASR unavailable: " + e.message + " — try ▶ Demo consult"); }
    S.pending--; renderTranscript();
  });
}

/* ------------------------------------------------------------------ transcript view */
const fullText = () => S.chunks.filter(Boolean).join(" ");
function renderTranscript() {
  const el = $("#transcript"), text = fullText();
  el.classList.toggle("placeholder", !text && !S.pending);
  if (!text && !S.pending) { el.textContent = "Ready — click Start Mic and speak naturally. VAD will auto-chunk your speech."; }
  else {
    let html = "", last = 0;
    for (const [a, b] of ranges(text, S.hl)) { html += esc(text.slice(last, a)) + "<mark>" + esc(text.slice(a, b)) + "</mark>"; last = b; }
    html += esc(text.slice(last)) + (S.pending ? ' <span class="pend">… transcribing</span>' : "");
    el.innerHTML = html; el.scrollTop = el.scrollHeight;
  }
  $("#words").textContent = text ? text.split(/\s+/).length : 0;
  $("#chunks").textContent = S.chunks.filter(Boolean).length;
}
function ranges(text, field) {
  if (!field || !S.summary) return [];
  const rs = [];
  for (const e of S.summary.evidence.filter(e => e.field === field)) {
    const w = e.quote.split(/[^A-Za-z0-9]+/).filter(Boolean).map(escRe);
    const m = w.length && new RegExp(w.join("[^A-Za-z0-9]+"), "i").exec(text);
    if (m) rs.push([m.index, m.index + m[0].length]);
  }
  rs.sort((a, b) => a[0] - b[0]);
  return rs.filter((r, i) => !i || r[0] >= rs[i - 1][1]);
}
function updateStats() {
  const c = S.capturedFrames; $("#saved").textContent = c ? Math.round(100 * (1 - S.sentFrames / c)) + "%" : "0%";
}

/* ------------------------------------------------------------------ summary view */
const list = (a, cls = "") => a.length ? a.map(x => `<span class="tag ${cls}">${esc(x)}</span>`).join("") : "";
const SECTIONS = [
  ["clinical_summary", "🗒 Clinical summary", s => `<textarea id="sumtxt" placeholder="AI-generated clinical summary will appear here...">${esc(s.clinical_summary)}</textarea>`],
  ["chief_complaint", "💬 Chief complaint", s => s.chief_complaint ? esc(s.chief_complaint) : ""],
  ["hpi", "📈 History of present illness", s => {
    const h = s.hpi, rows = [["Onset", h.onset], ["Duration", h.duration], ["Progression", h.progression]].filter(r => r[1]).map(r => `<div><span class="sub">${r[0]}:</span> ${esc(r[1])}</div>`).join("");
    return rows + (h.aggravating.length ? `<div><span class="sub">Aggravating:</span> ${esc(h.aggravating.join("; "))}</div>` : "") + (h.relieving.length ? `<div><span class="sub">Relieving:</span> ${esc(h.relieving.join("; "))}</div>` : "");
  }],
  ["symptoms", "🌡 Symptoms", s => list(s.symptoms.positive, "pos") + list(s.symptoms.negative.map(x => "no " + x), "neg")],
  ["past_medical_history", "↺ Past medical history", s => list(s.past_medical_history)],
  ["medications", "💊 Medications & allergies", s => s.medications.map(m => `<div>${esc([m.name, m.dose, m.frequency, m.adherence].filter(Boolean).join(" · "))}</div>`).join("") + (s.allergies.length ? `<div><span class="sub">Allergies:</span> ${list(s.allergies, "neg")}</div>` : "")],
  ["observations", "🔬 Examinations & tests", s => list(s.observations.vitals) + s.observations.exam.map(x => `<div>${esc(x)}</div>`).join("")],
  ["assessment", "🧠 Clinical impression", s => list(s.assessment)],
  ["plan", "📋 Plan & follow-up", s => {
    const p = s.plan, row = (l, a) => a.length ? `<div><span class="sub">${l}:</span> ${esc(a.join("; "))}</div>` : "";
    return row("Investigations", p.investigations) + row("Prescriptions", p.prescriptions) + row("Advice", p.advice) + (p.follow_up ? `<div><span class="sub">Follow-up:</span> ${esc(p.follow_up)}</div>` : "");
  }],
];
const NONE = { clinical_summary: "", chief_complaint: "No complaints extracted yet", hpi: "No history extracted yet", symptoms: "No symptoms extracted yet", past_medical_history: "No past medical history extracted", medications: "No medications extracted yet", observations: "No examinations extracted yet", assessment: "No diagnosis inferred", plan: "No plan extracted yet" };

function renderSummary(s) {
  S.summary = s;
  const p = s ? s.patient : { name: "", age: "", sex: "", identifiers: [] };
  $("#pname").textContent = "Patient Name: " + (p.name || "—");
  $("#pdetail").textContent = `Age: ${p.age || "—"}  Gender: ${p.sex || "—"}` + (p.identifiers.length ? "  ID: " + p.identifiers.join(", ") : "");
  $("#sections").innerHTML = SECTIONS.map(([k, title, fn]) => {
    const body = s ? fn(s) : ""; const filled = body && body !== "<textarea" ;
    return `<div class="sec" data-f="${k}"><h4>${title}</h4><div class="body">${body || `<div class="empty">${NONE[k]}</div>`}</div></div>`;
  }).join("");
  document.querySelectorAll(".sec").forEach(el => {
    el.onmouseenter = () => { S.hl = el.dataset.f; renderTranscript(); };
    el.onmouseleave = () => { S.hl = null; renderTranscript(); };
  });
  const m = s && s.meta;
  $("#grounding").innerHTML = m ? `Engine: <b>${esc(m.engine)}</b> · evidence grounded <b>${m.evidence_grounded}/${m.evidence_total}</b> quotes verified in transcript · hover a section to highlight its source` + (m.error ? ` · <span style="color:#be123c">LLM failed (${esc(m.error)}), used offline fallback</span>` : "") : "";
}

async function process() {
  const t = fullText(); if (!t) return toast("Nothing to process yet.");
  $("#process").disabled = true; $("#process").textContent = "Analysing…";
  try { renderSummary(await (await fetch("/api/analyze", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ transcript: t }) })).json()); }
  catch (e) { toast("Analysis failed: " + e.message); }
  $("#process").disabled = false; $("#process").textContent = "⚡ Process Transcript with AI";
}

/* ------------------------------------------------------------------ actions */
function toast(msg) { const t = document.createElement("div"); t.textContent = msg; t.style.cssText = "position:fixed;bottom:20px;left:50%;transform:translateX(-50%);background:#1e2a44;color:#fff;padding:10px 16px;border-radius:10px;font-size:13px;z-index:9"; document.body.append(t); setTimeout(() => t.remove(), 4000); }
function clearAll() { Object.assign(S, { chunks: [], pending: 0, speech: [], capturedFrames: 0, sentFrames: 0, hl: null }); renderTranscript(); renderSummary(null); updateStats(); }
function download(name, blob) { const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 2000); }
function wavBlob(chunks) {
  const n = chunks.reduce((a, c) => a + c.length, 0), b = new DataView(new ArrayBuffer(44 + n * 2));
  const w = (o, s) => [...s].forEach((c, i) => b.setUint8(o + i, c.charCodeAt(0)));
  w(0, "RIFF"); b.setUint32(4, 36 + n * 2, true); w(8, "WAVEfmt "); b.setUint32(16, 16, true); b.setUint16(20, 1, true); b.setUint16(22, 1, true);
  b.setUint32(24, 16000, true); b.setUint32(28, 32000, true); b.setUint16(32, 2, true); b.setUint16(34, 16, true); w(36, "data"); b.setUint32(40, n * 2, true);
  let o = 44; chunks.forEach(c => c.forEach(v => { b.setInt16(o, v, true); o += 2; }));
  return new Blob([b], { type: "audio/wav" });
}
function toRx(s) {
  const L = [`Patient: ${s.patient.name || "-"}, ${s.patient.age || "-"}, ${s.patient.sex || "-"}`, `C/O: ${s.chief_complaint}`,
    `Symptoms: ${s.symptoms.positive.join(", ")}${s.symptoms.negative.length ? " | Denies: " + s.symptoms.negative.join(", ") : ""}`,
    `PMH: ${s.past_medical_history.join("; ")}`, `Meds: ${s.medications.map(m => [m.name, m.dose, m.frequency].filter(Boolean).join(" ")).join("; ")}`,
    `Allergies: ${s.allergies.join(", ")}`, `O/E: ${[...s.observations.vitals, ...s.observations.exam].join("; ")}`,
    `Dx: ${s.assessment.join("; ")}`, `Rx: ${s.plan.prescriptions.join("; ")}`, `Investigations: ${s.plan.investigations.join(", ")}`,
    `Advice: ${s.plan.advice.join("; ")}`, `Follow-up: ${s.plan.follow_up}`, "", s.clinical_summary];
  return L.join("\n");
}

const DEMO = "Good morning, what brings you in today? Hi doctor, I'm Tushar Sharma, thirty years old, male. I've had a fever and a sore throat for three days. It gets worse at night and swallowing makes it hurt more. Paracetamol helps a little. I have no cough and no shortness of breath. I have type 2 diabetes and had my appendix removed in 2015. I take metformin 500 mg twice daily and I'm allergic to penicillin. Blood pressure is 120 over 80, temperature is 101 degrees fahrenheit, and the throat is red with swollen tonsils. This looks like acute tonsillitis. I'll order a throat swab and a CBC, and prescribe azithromycin 500 mg once daily for three days. Drink warm fluids and rest. Follow up in one week.";

$("#start").onclick = async () => {
  try { await startMic(); $("#start").disabled = true; $("#stop").disabled = false; }
  catch (e) { toast("Mic error: " + e.message); }
};
$("#stop").onclick = async () => { stopMic(); $("#start").disabled = false; $("#stop").disabled = true; await S.queue; if (fullText()) process(); };
$("#clear").onclick = clearAll;
$("#clearsum").onclick = () => renderSummary(null);
$("#process").onclick = process;
$("#save").onclick = () => {
  if (!fullText()) return toast("Nothing to save yet.");
  download("consultation.json", new Blob([JSON.stringify({ transcript: fullText(), summary: S.summary }, null, 2)], { type: "application/json" }));
  if (S.speech.length) download("speech-only.wav", wavBlob(S.speech));
};
$("#copy").onclick = async () => { if (!S.summary) return toast("Process a transcript first."); await navigator.clipboard.writeText(toRx(S.summary)); toast("Copied to clipboard"); };
$("#demo").onclick = async () => {
  clearAll(); $("#demo").disabled = true;
  for (const s of DEMO.split(/(?<=[.?!])\s+/)) { S.chunks.push(s); renderTranscript(); await new Promise(r => setTimeout(r, 650)); }
  $("#demo").disabled = false; process();
};

fetch("/api/health").then(r => r.json()).then(h => { $("#engines").innerHTML = `<span>ASR: ${esc(h.asr)}</span><span>LLM: ${esc(h.llm)}</span><span>VAD: in-browser</span>`; }).catch(() => { });
renderSummary(null);
