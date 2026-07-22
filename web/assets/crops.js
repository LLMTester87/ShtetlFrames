function escapeHtml(s) {
  return String(s ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function escapeAttr(s) {
  return escapeHtml(s).replace(/'/g, "&#39;");
}

function fmtTime(sec) {
  const s = Math.max(0, Math.round(Number(sec) || 0));
  const m = Math.floor(s / 60);
  const r = s % 60;
  return `${m}:${String(r).padStart(2, "0")}`;
}

function fmtBytes(n) {
  const b = Number(n) || 0;
  if (b < 1024) return `${b} B`;
  if (b < 1024 * 1024) return `${(b / 1024).toFixed(1)} KB`;
  return `${(b / (1024 * 1024)).toFixed(2)} MB`;
}

async function loadCrops() {
  const list = document.getElementById("cropsList");
  const summary = document.getElementById("cropsSummary");
  try {
    const res = await fetch("/api/crops");
    const data = await res.json();
    const crops = data.crops || [];
    const ready = crops.filter((c) => c.status === "ready").length;
    const queued = crops.filter((c) => c.status === "queued").length;
    summary.textContent =
      crops.length === 0
        ? "No crops yet. Open Review and click Generate crop on a hit."
        : `${ready} ready to download${queued ? ` · ${queued} generating` : ""}`;

    if (!crops.length) {
      list.innerHTML = `<div class="sheet placeholder">Nothing queued or ready</div>`;
      return;
    }

    list.innerHTML = crops
      .map((c) => {
        const name = c.crop_url
          ? String(c.crop_url).split("/").pop() || `cand_${c.id}_crop.jpg`
          : `cand_${c.id}_crop.jpg`;
        const title = c.video_id || `Hit #${c.id}`;
        const when =
          c.start_sec != null
            ? `${fmtTime(c.start_sec)} – ${fmtTime(c.end_sec)}`
            : "";
        if (c.status === "ready" && c.crop_url) {
          return `<article class="crop-card ready">
            <div class="crop-meta">
              <h2>${escapeHtml(title)}</h2>
              <p>#${c.id}${when ? ` · ${escapeHtml(when)}` : ""} · ${fmtBytes(c.bytes)}</p>
            </div>
            <a class="btn ok" href="${escapeAttr(c.crop_url)}" download="${escapeAttr(name)}">Download</a>
          </article>`;
        }
        if (c.status === "queued") {
          return `<article class="crop-card queued">
            <div class="crop-meta">
              <h2>${escapeHtml(title)}</h2>
              <p>#${c.id}${when ? ` · ${escapeHtml(when)}` : ""} · generating…</p>
            </div>
            <span class="crop-status">Queued</span>
          </article>`;
        }
        return `<article class="crop-card error">
          <div class="crop-meta">
            <h2>${escapeHtml(title)}</h2>
            <p>#${c.id} · ${escapeHtml(c.error || "failed")}</p>
          </div>
          <span class="crop-status">Error</span>
        </article>`;
      })
      .join("");
  } catch (e) {
    summary.textContent = "Could not load crops.";
    list.innerHTML = `<div class="sheet placeholder">${escapeHtml(e.message || e)}</div>`;
  }
}

document.getElementById("refreshCrops").addEventListener("click", () => loadCrops());
loadCrops();
setInterval(loadCrops, 4000);
