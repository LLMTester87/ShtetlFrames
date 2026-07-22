let pollTimer = null;
let searchTimer = null;
let settingsLoaded = false;
let buildPollTimer = null;
let queueState = {
  offset: 0,
  limit: 100,
  status: "",
  q: "",
  total: 0,
};

async function loadLanding() {
  const data = await fetch("/api/summary").then((r) => r.json());
  applyDiscoverDefaults(data);
  renderStats(data);
  renderJobs(data);
  updateScrapeEnabled(data);
  await loadQueuePage();
  if (!settingsLoaded) await loadSettingsForm();
}

function applyDiscoverDefaults(data) {
  const d = data.discover_defaults || {};
  const maxEl = document.getElementById("discoverMax");
  if (maxEl) {
    maxEl.min = "1";
    maxEl.max = String(d.hard_cap ?? 1_000_000);
    maxEl.step = "1";
    if (!maxEl.dataset.ready) {
      maxEl.value = String(d.default_max ?? 5000);
      maxEl.dataset.ready = "1";
    }
  }
  const scrapeMax = document.getElementById("maxN");
  if (scrapeMax) {
    scrapeMax.min = "1";
    scrapeMax.max = String(d.hard_cap ?? 1_000_000);
    scrapeMax.step = "1";
  }
  if (d.page_size && !document.getElementById("queuePageSize")?.dataset.ready) {
    const ps = document.getElementById("queuePageSize");
    if (ps) {
      ps.value = String(d.page_size);
      ps.dataset.ready = "1";
      queueState.limit = Number(d.page_size) || 100;
    }
  }
}

function renderStats(data) {
  const q = data.queue || {};
  const scrape = data.scrape || {};
  // Live scrape counters (not only rows that produced review hits).
  const videosDone = Number(
    scrape.completed ?? q.n_done ?? data.videos_scanned ?? 0
  );
  const hitSegs = Number(data.n_candidates ?? 0);
  const scrapeHits = Number(scrape.hits ?? 0);
  const scrapeTotal = Number(scrape.total ?? q.n_queue ?? 0);
  const stats = document.getElementById("stats");
  if (stats) {
    const scannedLabel =
      scrape.status === "running" && scrapeTotal > 0
        ? `Videos scanned (${videosDone}/${scrapeTotal})`
        : "Videos scanned";
    const candLabel =
      scrape.status === "running" && scrapeHits > 0 && hitSegs === 0
        ? `Hit segments (this run: ${scrapeHits})`
        : "Candidates";
    stats.innerHTML = `
      <div class="stat"><strong>${hitSegs}</strong><span>${candLabel}</span></div>
      <div class="stat"><strong>${videosDone}</strong><span>${scannedLabel}</span></div>
      <div class="stat"><strong>${data.n_pending ?? 0}</strong><span>Pending review</span></div>
      <div class="stat"><strong>${data.n_accepted ?? 0}</strong><span>Accepted</span></div>
    `;
  }
  const queueStats = document.getElementById("queueStats");
  if (queueStats) {
    queueStats.innerHTML = `
      <div class="stat"><strong>${q.n_queue ?? 0}</strong><span>In queue</span></div>
      <div class="stat"><strong>${q.n_pending ?? 0}</strong><span>Pending scrape</span></div>
      <div class="stat"><strong>${q.n_active ?? 0}</strong><span>Active now</span></div>
      <div class="stat"><strong>${q.n_done ?? 0}</strong><span>Done</span></div>
      <div class="stat"><strong>${q.n_error ?? 0}</strong><span>Errors</span></div>
    `;
  }
  const hint = document.getElementById("scanBackendHint");
  if (hint) {
    const s = data.scan || {};
    if (s.backend === "runpod") {
      hint.textContent = `Scan backend: RunPod (API key only) · ${s.gpu_type || "GPU"} · up to ${s.max_inflight ?? 8} parallel.`;
    } else if (s.requested === "runpod" && !s.runpod_configured) {
      hint.textContent =
        "Scan backend: RunPod — paste your API key in Settings to spin a cloud GPU.";
    } else {
      hint.textContent = "Scan backend: RunPod — paste API key in Settings.";
    }
  }
  const workersEl = document.getElementById("workers");
  if (workersEl && data.scan) {
    const maxW = data.scan.backend === "runpod" ? Number(data.scan.max_inflight) || 8 : 8;
    workersEl.max = String(maxW);
    if (!workersEl.dataset.ready) {
      workersEl.value = String(data.scan.backend === "runpod" ? maxW : 2);
      workersEl.dataset.ready = "1";
    }
  }
}

function renderJobs(data) {
  const d = data.discover || {};
  const s = data.scrape || {};
  const dProg = document.getElementById("discoverProgress");
  const dBar = document.getElementById("discoverBar");
  const dMsg = document.getElementById("discoverMsg");
  const discovering = d.status === "running";
  if (dProg) {
    dProg.hidden = !(discovering || d.status === "done" || d.status === "error");
    if (dBar) dBar.style.width = `${Number(d.progress) || 0}%`;
    if (dMsg) dMsg.textContent = d.message || d.status || "";
    dProg.className = `job-status ${d.status || "idle"}`;
  }
  const discoverBtn = document.getElementById("discoverBtn");
  if (discoverBtn) discoverBtn.disabled = discovering || s.status === "running";

  const sProg = document.getElementById("scrapeProgress");
  const sBar = document.getElementById("scrapeBar");
  const sMsg = document.getElementById("scrapeMsg");
  const scraping = s.status === "running";
  if (sProg) {
    sProg.hidden = !(scraping || s.status === "done" || s.status === "error");
    if (sBar) sBar.style.width = `${Number(s.progress) || 0}%`;
    if (sMsg) {
      sMsg.textContent = s.message || s.status || "";
    }
    sProg.className = `job-status ${s.status || "idle"}`;
  }

  if (discovering || scraping) startPolling(scraping ? 1000 : 2000);
  else stopPolling();
}

function updateScrapeEnabled(data) {
  const btn = document.getElementById("scrapeBtn");
  if (!btn) return;
  const pending = (data.queue && data.queue.n_pending) || 0;
  const errors = (data.queue && data.queue.n_error) || 0;
  const ready = pending + errors;
  const busy =
    (data.discover && data.discover.status === "running") ||
    (data.scrape && data.scrape.status === "running");
  btn.disabled = busy || ready === 0;
  btn.title =
    ready === 0
      ? "No videos ready — Discover first"
      : busy
        ? "A job is already running"
        : errors > 0 && pending === 0
          ? "Retry failed videos"
          : errors > 0
            ? "Start scrape (includes retrying errors)"
            : "Start download + scan";
  if (!busy && errors > 0 && pending === 0) {
    btn.textContent = "Retry failed";
  } else if (!busy) {
    btn.textContent = "Start scrape";
  }
}

function queueQuery() {
  const p = new URLSearchParams({
    offset: String(queueState.offset),
    limit: String(queueState.limit),
  });
  if (queueState.status) p.set("status", queueState.status);
  if (queueState.q) p.set("q", queueState.q);
  return p.toString();
}

async function loadQueuePage() {
  const el = document.getElementById("queueTableBody");
  const meta = document.getElementById("queuePageMeta");
  if (!el) return;
  let data;
  try {
    data = await fetch(`/api/queue?${queueQuery()}`).then((r) => r.json());
  } catch {
    el.innerHTML = `<tr><td colspan="6" class="empty-cell">Could not load queue</td></tr>`;
    return;
  }
  queueState.total = data.total || 0;
  queueState.offset = data.offset ?? queueState.offset;
  queueState.limit = data.limit ?? queueState.limit;
  const items = data.items || [];
  if (!items.length) {
    el.innerHTML = `<tr><td colspan="6" class="empty-cell">Queue empty — discover a URL first.</td></tr>`;
  } else {
    const frag = document.createDocumentFragment();
    const tmp = document.createElement("tbody");
    tmp.innerHTML = items
      .map(
        (r) => `
      <tr>
        <td class="col-id">${r.id}</td>
        <td><span class="status-pill status-${escapeAttr(r.status || "pending")}">${escapeHtml(r.status || "—")}</span></td>
        <td class="col-title">
          <a href="${escapeAttr(r.url)}" target="_blank" rel="noopener">${escapeHtml(r.title || r.url)}</a>
          ${r.detail ? `<div class="row-detail">${escapeHtml(String(r.detail).slice(0, 220))}</div>` : ""}
          ${r.error ? `<div class="row-error">${escapeHtml(String(r.error).slice(0, 200))}</div>` : ""}
        </td>
        <td class="col-source">${escapeHtml(r.source || "—")}</td>
        <td class="col-year">${escapeHtml(r.year || "—")}</td>
        <td class="col-act"><button type="button" class="btn danger small delete-btn" data-url="${escapeAttr(r.url)}">Delete</button></td>
      </tr>`
      )
      .join("");
    while (tmp.firstChild) frag.appendChild(tmp.firstChild);
    el.replaceChildren(frag);
    el.querySelectorAll(".delete-btn").forEach((btn) => {
      btn.addEventListener("click", () => deleteSource(btn.dataset.url));
    });
  }
  const start = queueState.total ? queueState.offset + 1 : 0;
  const end = Math.min(queueState.offset + queueState.limit, queueState.total);
  if (meta) {
    meta.textContent = queueState.total
      ? `Showing ${start.toLocaleString()}–${end.toLocaleString()} of ${queueState.total.toLocaleString()}`
      : "0 discovered";
  }
  const prev = document.getElementById("queuePrev");
  const next = document.getElementById("queueNext");
  if (prev) prev.disabled = queueState.offset <= 0;
  if (next) next.disabled = queueState.offset + queueState.limit >= queueState.total;
}

async function discover(ev) {
  ev.preventDefault();
  const url = document.getElementById("sourceUrl").value.trim();
  let max_items = Number(document.getElementById("discoverMax")?.value);
  if (!Number.isFinite(max_items) || max_items < 1) max_items = 5000;
  max_items = Math.min(1_000_000, Math.floor(max_items));
  const msg = document.getElementById("sourceFormMsg");
  msg.textContent = "Starting discovery…";
  let res;
  try {
    res = await fetch("/api/discover", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url, max_items }),
    });
  } catch (e) {
    msg.textContent = "Cannot reach server — is start_web.bat running?";
    return;
  }
  let data = {};
  try {
    data = await res.json();
  } catch {
    msg.textContent = `Server error (${res.status}). Restart start_web.bat.`;
    return;
  }
  if (!res.ok || !data.ok) {
    msg.textContent =
      data.error === "busy"
        ? "Already busy."
        : data.error === "not found"
          ? "API missing — restart start_web.bat to load the latest server."
          : data.error || data.message || `Discover failed (${res.status})`;
    return;
  }
  msg.textContent = `Discovering (max ${max_items.toLocaleString()})…`;
  queueState.offset = 0;
  startPolling();
  await loadLanding();
}

async function startScrape() {
  const mode = document.querySelector('input[name="maxMode"]:checked')?.value || "all";
  const max_videos = mode === "all" ? "all" : Number(document.getElementById("maxN").value) || 25;
  const workers = Number(document.getElementById("workers").value) || 2;
  const msg = document.getElementById("scrapeMsg");
  const sProg = document.getElementById("scrapeProgress");
  const logEl = document.getElementById("runpodSetupLog");
  if (sProg) sProg.hidden = false;

  let settings;
  try {
    settings = await collectSettingsPayload();
    settings.SCAN_BACKEND = "runpod";
  } catch (e) {
    alert(String(e.message || e));
    return;
  }
  if (msg) msg.textContent = "Preparing cloud GPU (save → pod → scrape)…";
  if (logEl) {
    logEl.hidden = false;
    logEl.textContent = "Starting…\n";
  }
  let res;
  try {
    res = await fetch("/api/runpod/go", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ settings, max_videos, workers }),
    });
  } catch (e) {
    alert(
      "Could not start scrape (connection dropped). " +
        "If the ShtetlFrames window is open, hard-refresh this page (Ctrl+F5). " +
        "Otherwise run start_web.bat. " +
        String(e && e.message ? e.message : "")
    );
    return;
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok || !data.ok) {
    alert(data.error || data.message || `RunPod start failed (${res.status})`);
    return;
  }
  if (buildPollTimer) clearInterval(buildPollTimer);
  buildPollTimer = setInterval(pollRunpodGo, 1500);
  await pollRunpodGo();
  startPolling();
}

async function pollRunpodGo() {
  const msg = document.getElementById("scrapeMsg");
  const logEl = document.getElementById("runpodSetupLog");
  const data = await fetch("/api/runpod/go").then((r) => r.json()).catch(() => ({}));
  const job = data.job || {};
  if (logEl) {
    logEl.hidden = false;
    logEl.textContent = job.log || job.message || "";
    logEl.scrollTop = logEl.scrollHeight;
  }
  if (msg) msg.textContent = job.message || job.status || "";
  if (job.status === "done" || job.status === "error" || job.status === "idle") {
    if (buildPollTimer) {
      clearInterval(buildPollTimer);
      buildPollTimer = null;
    }
    if (job.status === "error") {
      alert(job.message || "RunPod setup failed");
    }
    settingsLoaded = false;
    await loadSettingsForm();
    await loadLanding();
    startPolling();
  }
}


async function deleteSource(url) {
  if (!confirm("Remove from queue?")) return;
  await fetch("/api/queue/delete", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ url }),
  });
  await loadLanding();
}

async function clearQueue() {
  if (!confirm("Clear the entire discover queue?")) return;
  await fetch("/api/queue/clear", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: "{}",
  });
  queueState.offset = 0;
  await loadLanding();
}

function startPolling(ms = 2000) {
  if (pollTimer) {
    if (pollTimer._ms === ms) return;
    clearInterval(pollTimer);
    pollTimer = null;
  }
  pollTimer = setInterval(loadLanding, ms);
  pollTimer._ms = ms;
}

function stopPolling() {
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function escapeAttr(s) {
  return escapeHtml(s).replaceAll("'", "&#39;");
}

document.querySelectorAll('input[name="maxMode"]').forEach((el) => {
  el.addEventListener("change", () => {
    const nChecked = document.querySelector('input[name="maxMode"][value="n"]').checked;
    document.getElementById("maxN").disabled = !nChecked;
  });
});

document.getElementById("queueSearch")?.addEventListener("input", (ev) => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => {
    queueState.q = ev.target.value.trim();
    queueState.offset = 0;
    loadQueuePage();
  }, 250);
});

document.getElementById("queueStatus")?.addEventListener("change", (ev) => {
  queueState.status = ev.target.value;
  queueState.offset = 0;
  loadQueuePage();
});

document.getElementById("queuePageSize")?.addEventListener("change", (ev) => {
  queueState.limit = Math.min(500, Math.max(25, Number(ev.target.value) || 100));
  queueState.offset = 0;
  loadQueuePage();
});

document.getElementById("queuePrev")?.addEventListener("click", () => {
  queueState.offset = Math.max(0, queueState.offset - queueState.limit);
  loadQueuePage();
});

document.getElementById("queueNext")?.addEventListener("click", () => {
  if (queueState.offset + queueState.limit < queueState.total) {
    queueState.offset += queueState.limit;
    loadQueuePage();
  }
});

document.getElementById("discoverForm").addEventListener("submit", discover);
document.getElementById("scrapeBtn").addEventListener("click", startScrape);
document.getElementById("clearQueueBtn").addEventListener("click", clearQueue);
document.getElementById("settingsForm")?.addEventListener("submit", saveSettings);
document.getElementById("cookiesRefreshBtn")?.addEventListener("click", async () => {
  const msg = document.getElementById("settingsMsg");
  if (msg) msg.textContent = "Exporting YouTube cookies from your browser…";
  try {
    const res = await fetch("/api/youtube/cookies", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ force: true }),
    });
    const data = await res.json();
    if (msg) {
      msg.textContent = data.ok
        ? `Cookies ready (${data.bytes || 0} bytes). Start scrape to use them on pods.`
        : data.message || data.error || "Cookie export failed — sign into YouTube in that browser.";
    }
  } catch {
    if (msg) msg.textContent = "Cookie export failed — is the server running?";
  }
});

document.getElementById("cookiesHarInput")?.addEventListener("change", async (ev) => {
  const input = ev.target;
  const file = input?.files?.[0];
  const msg = document.getElementById("settingsMsg");
  if (!file) return;
  if (msg) msg.textContent = `Reading HAR (${file.name})…`;
  try {
    const text = await file.text();
    let har;
    try {
      har = JSON.parse(text);
    } catch {
      if (msg) {
        msg.textContent =
          "Not a valid HAR JSON file. In Edge: Network panel → Save all as HAR with content.";
      }
      return;
    }
    if (msg) msg.textContent = "Importing YouTube cookies from HAR…";
    const body = har && typeof har === "object" && har.log ? har : { har };
    const res = await fetch("/api/youtube/cookies/har", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (msg) {
      msg.textContent = data.ok
        ? `${data.message || "HAR cookies imported"}. Start scrape to use them on pods.`
        : data.message || data.error || "HAR import failed.";
    }
  } catch {
    if (msg) msg.textContent = "HAR import failed — is the server running? File may be too large.";
  } finally {
    input.value = "";
  }
});
document.getElementById("podStopBtn")?.addEventListener("click", stopGpuPod);
loadLanding();

async function collectSettingsPayload() {
  const form = document.getElementById("settingsForm");
  const payload = {};
  const clearSecrets = [];
  form.querySelectorAll("[name]").forEach((el) => {
    const key = el.name;
    if (!key) return;
    if (el.dataset.secret === "1") {
      if (el.dataset.clear === "1") {
        clearSecrets.push(key);
        payload[key] = "";
        return;
      }
      const v = el.value.trim();
      if (!v) return;
      payload[key] = v;
      return;
    }
    payload[key] = el.value;
  });
  if (clearSecrets.length) payload._clear_secrets = clearSecrets;
  return payload;
}

async function saveSettingsQuiet() {
  const payload = await collectSettingsPayload();
  const res = await fetch("/api/settings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok || !data.ok) throw new Error(data.error || `Save failed (${res.status})`);
  settingsLoaded = false;
  await loadSettingsForm();
  return data;
}

async function stopGpuPod() {
  const msg = document.getElementById("settingsMsg");
  if (msg) msg.textContent = "Stopping GPU Pod…";
  const res = await fetch("/api/runpod/pod/stop", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: "{}",
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok || !data.ok) {
    if (msg) msg.textContent = data.error || `Stop failed (${res.status})`;
    return;
  }
  if (msg) msg.textContent = "GPU Pod stopped.";
}

async function saveSettings(ev) {
  ev.preventDefault();
  const msg = document.getElementById("settingsMsg");
  if (msg) msg.textContent = "Saving…";
  try {
    await saveSettingsQuiet();
    if (msg) msg.textContent = "Saved.";
    await loadLanding();
  } catch (e) {
    if (msg) msg.textContent = String(e.message || e);
  }
}

async function loadSettingsForm() {
  const host = document.getElementById("settingsFields");
  if (!host) return;
  let data;
  try {
    data = await fetch("/api/settings").then((r) => r.json());
  } catch {
    host.innerHTML = `<p class="job-hint">Could not load settings</p>`;
    return;
  }
  const fields = data.fields || [];
  let lastSection = null;
  host.innerHTML = fields
    .map((f) => {
      const id = `setting_${f.key}`;
      const help = f.help ? `<span class="settings-help">${escapeHtml(f.help)}</span>` : "";
      const section = (f.section || "").trim();
      let heading = "";
      if (section && section !== lastSection) {
        lastSection = section;
        heading = `<div class="settings-section">${escapeHtml(section)}</div>`;
      }
      const vis = Array.isArray(f.visible_for) ? f.visible_for : null;
      const visKey = (f.visible_for_key || "PROXY_PROVIDER").trim() || "PROXY_PROVIDER";
      const wrapOpen = vis
        ? `<div class="settings-conditional" data-visible-for="${escapeAttr(vis.join(","))}" data-visible-key="${escapeAttr(visKey)}">`
        : `<div class="settings-field">`;
      const wrapClose = `</div>`;
      if (f.type === "select") {
        const labels = f.option_labels || {};
        const opts = (f.options || [])
          .map((o) => {
            const label = labels[o] || o;
            return `<option value="${escapeAttr(o)}"${o === f.value ? " selected" : ""}>${escapeHtml(label)}</option>`;
          })
          .join("");
        return `${heading}${wrapOpen}<label class="settings-row" for="${id}"><span>${escapeHtml(f.label)}</span>
          <select id="${id}" name="${escapeAttr(f.key)}">${opts}</select>${help}</label>${wrapClose}`;
      }
      if (f.type === "password") {
          const ph = f.has_value
            ? `•••• saved — paste new to replace`
            : `enter ${escapeAttr(f.label.toLowerCase())}`;
          return `${heading}${wrapOpen}<label class="settings-row" for="${id}"><span>${escapeHtml(f.label)}</span>
          <input id="${id}" name="${escapeAttr(f.key)}" type="password" autocomplete="off" placeholder="${ph}" data-secret="1" data-has="${f.has_value ? "1" : "0"}" />
          ${f.has_value ? `<button type="button" class="btn ghost small clear-secret" data-key="${escapeAttr(f.key)}">Clear</button>` : ""}
          ${help}</label>${wrapClose}`;
      }
      const extra =
        f.type === "number"
          ? ` type="number" min="${f.min ?? ""}" max="${f.max ?? ""}" step="${f.step ?? "1"}"`
          : ` type="text"`;
      return `${heading}${wrapOpen}<label class="settings-row" for="${id}"><span>${escapeHtml(f.label)}</span>
        <input id="${id}" name="${escapeAttr(f.key)}"${extra} value="${escapeAttr(f.value ?? "")}" />
        ${help}</label>${wrapClose}`;
    })
    .join("");
  host.querySelectorAll(".clear-secret").forEach((btn) => {
    btn.addEventListener("click", () => {
      const inp = document.getElementById(`setting_${btn.dataset.key}`);
      if (inp) {
        inp.value = "";
        inp.dataset.clear = "1";
        inp.placeholder = "will clear on Save";
      }
    });
  });
  const applyConditionalVisibility = () => {
    host.querySelectorAll(".settings-conditional").forEach((el) => {
      const key = (el.dataset.visibleKey || "PROXY_PROVIDER").trim() || "PROXY_PROVIDER";
      const sel = document.getElementById(`setting_${key}`);
      const cur = (sel?.value || "").trim().toLowerCase();
      const allowed = (el.dataset.visibleFor || "")
        .split(",")
        .map((s) => s.trim().toLowerCase())
        .filter(Boolean);
      el.hidden = allowed.length > 0 && !allowed.includes(cur);
    });
  };
  host.querySelectorAll("select").forEach((sel) => {
    sel.addEventListener("change", applyConditionalVisibility);
  });
  applyConditionalVisibility();
  settingsLoaded = true;
}
