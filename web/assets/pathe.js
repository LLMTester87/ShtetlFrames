(() => {
  let pathePollTimer = null;
  let patheQueueState = { offset: 0, limit: 100, status: "", q: "", total: 0 };

  async function loadPathe() {
    const data = await fetch("/api/pathe/summary").then((r) => r.json());
    renderPatheStats(data);
    renderPatheJobs(data);
    await loadPatheQueue();
  }

  function renderPatheStats(data) {
    const q = data.queue || {};
    const el = document.getElementById("patheStats");
    if (!el) return;
    el.innerHTML = `
      <div class="stat"><strong>${q.n_queue ?? 0}</strong><span>In queue</span></div>
      <div class="stat"><strong>${q.n_pending ?? 0}</strong><span>Pending</span></div>
      <div class="stat"><strong>${q.n_active ?? 0}</strong><span>Active</span></div>
      <div class="stat"><strong>${q.n_done ?? 0}</strong><span>Done</span></div>
      <div class="stat"><strong>${q.n_error ?? 0}</strong><span>Errors</span></div>
    `;
  }

  function renderPatheJobs(data) {
    const d = data.discover || {};
    const s = data.scrape || {};
    const discovering = d.status === "running";
    const scraping = s.status === "running";

    const dProg = document.getElementById("patheDiscoverProgress");
    const dBar = document.getElementById("patheDiscoverBar");
    const dMsg = document.getElementById("patheDiscoverMsg");
    if (dProg) {
      dProg.hidden = !(discovering || d.status === "done" || d.status === "error");
      if (dBar) dBar.style.width = `${Number(d.progress) || 0}%`;
      if (dMsg) dMsg.textContent = d.message || d.status || "";
      dProg.className = `job-status ${d.status || "idle"}`;
    }

    const sProg = document.getElementById("patheScrapeProgress");
    const sBar = document.getElementById("patheScrapeBar");
    const sMsg = document.getElementById("patheScrapeMsg");
    if (sProg) {
      sProg.hidden = !(scraping || s.status === "done" || s.status === "error");
      if (sBar) sBar.style.width = `${Number(s.progress) || 0}%`;
      if (sMsg) {
        const live = (data.live || [])
          .slice(0, 4)
          .map((x) => `${(x.title || "").slice(0, 40)}: ${x.detail || x.phase || ""}`)
          .join("\n");
        sMsg.textContent = [s.message || s.status || "", live].filter(Boolean).join("\n");
      }
      sProg.className = `job-status ${s.status || "idle"}`;
    }

    const ds = document.getElementById("patheDiscoverStatus");
    const ss = document.getElementById("patheScrapeStatus");
    if (ds) {
      ds.textContent = discovering
        ? `${d.message || "Discovering…"} (${Math.round(d.progress || 0)}%)`
        : d.message || d.status || "Idle.";
    }
    if (ss) {
      ss.textContent = scraping
        ? s.message || "Scraping…"
        : s.message || s.status || "Idle.";
    }

    for (const id of ["patheDiscoverBtn", "patheDiscoverAllBtn"]) {
      const btn = document.getElementById(id);
      if (btn) btn.disabled = discovering;
    }
    const scrapeBtn = document.getElementById("patheScrapeBtn");
    if (scrapeBtn) scrapeBtn.disabled = scraping;

    if (discovering || scraping) {
      if (!pathePollTimer) {
        pathePollTimer = setInterval(loadPathe, scraping ? 1000 : 2000);
      }
    }
  }

  function assetIdFromUrl(url) {
    const m = String(url || "").match(/\/asset\/(\d+)/i);
    return m ? m[1] : "—";
  }

  async function loadPatheQueue() {
    const body = document.getElementById("patheQueueBody");
    const meta = document.getElementById("patheQueueMeta");
    if (!body) return;
    const params = new URLSearchParams({
      offset: String(patheQueueState.offset),
      limit: String(patheQueueState.limit),
    });
    if (patheQueueState.status) params.set("status", patheQueueState.status);
    if (patheQueueState.q) params.set("q", patheQueueState.q);
    let data;
    try {
      data = await fetch("/api/pathe/queue?" + params).then((r) => r.json());
    } catch {
      body.innerHTML = `<tr><td colspan="5" class="empty-cell">Could not load queue</td></tr>`;
      return;
    }
    patheQueueState.total = data.total || 0;
    const items = data.items || [];
    if (!items.length) {
      body.innerHTML = `<tr><td colspan="5" class="empty-cell">Queue empty — Discover all to begin.</td></tr>`;
    } else {
      body.innerHTML = items
        .map((r) => {
          const aid = assetIdFromUrl(r.url);
          return `
        <tr>
          <td class="col-id">${r.id}</td>
          <td><span class="status-pill status-${escapeAttr(r.status || "pending")}">${escapeHtml(r.status || "—")}</span></td>
          <td class="col-title">
            <a href="${escapeAttr(r.url)}" target="_blank" rel="noopener">${escapeHtml(r.title || "Asset " + aid)}</a>
            ${r.error ? `<div class="row-error">${escapeHtml(String(r.error).slice(0, 180))}</div>` : ""}
            ${r.detail ? `<div class="row-detail">${escapeHtml(String(r.detail).slice(0, 180))}</div>` : ""}
          </td>
          <td class="col-year">${escapeHtml(aid)}</td>
          <td class="col-act"></td>
        </tr>`;
        })
        .join("");
    }
    const start = patheQueueState.total ? patheQueueState.offset + 1 : 0;
    const end = Math.min(
      patheQueueState.offset + patheQueueState.limit,
      patheQueueState.total
    );
    if (meta) {
      meta.textContent = patheQueueState.total
        ? `Showing ${start.toLocaleString()}–${end.toLocaleString()} of ${patheQueueState.total.toLocaleString()}`
        : "0 discovered";
    }
    const prev = document.getElementById("pathePrev");
    const next = document.getElementById("patheNext");
    if (prev) prev.disabled = patheQueueState.offset <= 0;
    if (next) {
      next.disabled =
        patheQueueState.offset + patheQueueState.limit >= patheQueueState.total;
    }
  }

  function escapeHtml(s) {
    return String(s || "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }
  function escapeAttr(s) {
    return escapeHtml(s).replace(/"/g, "&quot;");
  }

  async function postDiscover(body) {
    const payload = {
      auto_scrape: true,
      workers: Number(document.getElementById("patheWorkers")?.value || 8),
      ...body,
    };
    const res = await fetch("/api/pathe/discover", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!data.ok) alert(data.error || "Discover failed");
    patheQueueState.offset = 0;
    await loadPathe();
  }

  document.getElementById("patheDiscoverForm")?.addEventListener("submit", async (e) => {
    e.preventDefault();
    await postDiscover({
      query: document.getElementById("patheQuery")?.value || "",
      max_items: Number(document.getElementById("patheMax")?.value || 5000),
    });
  });

  document.getElementById("patheDiscoverAllBtn")?.addEventListener("click", async () => {
    await postDiscover({
      all: true,
      query: "",
      max_items: Number(document.getElementById("patheMax")?.value || 5000),
    });
  });

  document.getElementById("patheScrapeBtn")?.addEventListener("click", async () => {
    const maxRaw = (document.getElementById("patheScrapeMax")?.value || "all").trim();
    const res = await fetch("/api/pathe/scrape", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        max_videos: maxRaw,
        workers: Number(document.getElementById("patheWorkers")?.value || 8),
      }),
    });
    const data = await res.json();
    if (!data.ok) alert(data.error || data.job?.message || "Scrape failed");
    await loadPathe();
  });

  document.getElementById("patheClearBtn")?.addEventListener("click", async () => {
    if (!confirm("Clear all British Pathé rows from the queue?")) return;
    await fetch("/api/pathe/queue/clear", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: "{}",
    });
    patheQueueState.offset = 0;
    await loadPathe();
  });

  document.getElementById("patheQueueSearch")?.addEventListener("input", (e) => {
    patheQueueState.q = e.target.value || "";
    patheQueueState.offset = 0;
    clearTimeout(window._patheSearchT);
    window._patheSearchT = setTimeout(loadPatheQueue, 250);
  });

  document.getElementById("patheQueueStatus")?.addEventListener("change", (e) => {
    patheQueueState.status = e.target.value || "";
    patheQueueState.offset = 0;
    loadPatheQueue();
  });

  document.getElementById("pathePageSize")?.addEventListener("change", (e) => {
    patheQueueState.limit = Number(e.target.value) || 100;
    patheQueueState.offset = 0;
    loadPatheQueue();
  });

  document.getElementById("pathePrev")?.addEventListener("click", () => {
    patheQueueState.offset = Math.max(0, patheQueueState.offset - patheQueueState.limit);
    loadPatheQueue();
  });
  document.getElementById("patheNext")?.addEventListener("click", () => {
    patheQueueState.offset += patheQueueState.limit;
    loadPatheQueue();
  });

  if (document.getElementById("patheDiscoverForm") || document.getElementById("patheStats")) {
    loadPathe();
    pathePollTimer = setInterval(loadPathe, 2000);
  }
})();
