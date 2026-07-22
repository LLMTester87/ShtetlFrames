let pollTimer = null;

function escapeHtml(s) {
  return String(s ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function fmtTs(sec) {
  const n = Number(sec);
  if (!Number.isFinite(n) || n <= 0) return "—";
  try {
    return new Date(n * 1000).toLocaleTimeString();
  } catch {
    return "—";
  }
}

function jobLine(label, j) {
  if (!j || j.status === "none" || !j.status) {
    return `<div class="health-job"><span>${escapeHtml(label)}</span><strong class="muted">idle</strong></div>`;
  }
  const bits = [
    j.status,
    j.workers != null ? `${j.workers}w` : "",
    j.completed != null ? `${j.completed}${j.total != null ? "/" + j.total : ""}` : "",
  ].filter(Boolean);
  return `<div class="health-job">
    <span>${escapeHtml(label)}</span>
    <strong>${escapeHtml(bits.join(" · "))}</strong>
    <em>${escapeHtml(j.message || "")}</em>
  </div>`;
}

function render(data) {
  const alertsEl = document.getElementById("healthAlerts");
  const alerts = data.alerts || [];
  if (!alerts.length) {
    alertsEl.innerHTML = `<div class="health-alert ok">All clear</div>`;
  } else {
    alertsEl.innerHTML = alerts
      .map(
        (a) =>
          `<div class="health-alert ${escapeHtml(a.level)}">${escapeHtml(a.message)}</div>`
      )
      .join("");
  }

  const s = data.summary || {};
  const q = (data.queue && data.queue.pathe) || {};
  document.getElementById("healthStats").innerHTML = `
    <div class="stat"><span>Pods</span><strong>${s.busy_count ?? 0} busy · ${s.healthy_count ?? 0}/${s.pod_count ?? 0} healthy</strong></div>
    <div class="stat"><span>Idle healthy</span><strong>${s.idle_healthy_count ?? 0}</strong></div>
    <div class="stat"><span>Ollama ready</span><strong>${s.ollama_ready_count ?? 0}/${s.pod_count ?? 0}</strong></div>
    <div class="stat"><span>Scrape pool</span><strong>${s.scrape_pool_size ?? 0}</strong></div>
    <div class="stat"><span>Stack</span><strong>${s.pathe_stack ?? "—"} / ${s.pathe_stack_max ?? "—"}</strong></div>
    <div class="stat"><span>MAX_INFLIGHT</span><strong>${s.max_inflight ?? "—"}</strong></div>
  `;

  const jobs = data.jobs || {};
  document.getElementById("healthJobs").innerHTML = [
    jobLine("Pathé scrape", jobs.pathe_scrape),
    jobLine("Pathé discover", jobs.pathe_discover),
    jobLine("YT scrape", jobs.scrape),
    jobLine("YT discover", jobs.discover),
  ].join("");

  document.getElementById("healthQueue").innerHTML = `
    <div class="stat"><span>Pending</span><strong>${q.n_pending ?? 0}</strong></div>
    <div class="stat"><span>Active</span><strong>${q.n_active ?? 0}</strong></div>
    <div class="stat"><span>Done</span><strong>${q.n_done ?? 0}</strong></div>
    <div class="stat"><span>Error</span><strong>${q.n_error ?? 0}</strong></div>
  `;

  const tbody = document.getElementById("healthPods");
  const pods = data.pods || [];
  if (!pods.length) {
    tbody.innerHTML = `<tr><td colspan="7" class="empty-cell">No pods found</td></tr>`;
  } else {
    tbody.innerHTML = pods
      .map((p) => {
        const status = !p.healthy
          ? `<span class="badge reject">down</span>`
          : p.busy
            ? `<span class="badge accept">busy</span>`
            : `<span class="badge pending">idle</span>`;
        const inf =
          p.inflight != null
            ? `${p.inflight}${p.inflight_limit_pathe != null ? "/" + p.inflight_limit_pathe : ""}`
            : "—";
        let ollama = "—";
        if (p.ollama_model_ready === true) {
          ollama = `<span class="badge accept">ready</span>`;
        } else if (p.ollama_pulling) {
          ollama = `<span class="badge pending">pulling</span>`;
        } else if (p.ollama_ready === true) {
          ollama = `<span class="badge pending">no model</span>`;
        } else if (p.ollama_ready === false) {
          ollama = `<span class="badge reject">down</span>`;
        }
        const err = p.error || p.sync_error || "";
        return `<tr class="${p.busy ? "row-busy" : p.healthy ? "" : "row-down"}">
          <td class="col-title">${escapeHtml(p.name || "?")}</td>
          <td>${status}</td>
          <td>${escapeHtml(p.phase || "—")}</td>
          <td>${escapeHtml(inf)}</td>
          <td title="${escapeHtml(p.ollama_model || "")}">${ollama}</td>
          <td class="col-title">${escapeHtml(p.title || p.message || "")}</td>
          <td class="col-source">${escapeHtml(err)}</td>
        </tr>`;
      })
      .join("");
  }

  document.getElementById("healthMeta").textContent =
    `${fmtTs(data.ts)} · ${data.probe_ms ?? "—"}ms`;
}

async function loadHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    if (!data.ok && data.error) {
      document.getElementById("healthAlerts").innerHTML =
        `<div class="health-alert red">${escapeHtml(data.error)}</div>`;
      return;
    }
    render(data);
  } catch (e) {
    document.getElementById("healthAlerts").innerHTML =
      `<div class="health-alert red">Failed to load health: ${escapeHtml(e.message || e)}</div>`;
  }
}

function startPolling() {
  if (pollTimer) clearInterval(pollTimer);
  loadHealth();
  pollTimer = setInterval(loadHealth, 3000);
}

document.getElementById("refreshHealth")?.addEventListener("click", () => loadHealth());
startPolling();
