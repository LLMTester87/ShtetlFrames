let allCandidates = [];
let sources = [];
let activeVideoId = null;
let hitIndex = 0;
let status = "";
let query = "";
let cropPollTimer = null;
let sortKey = "scored";
let sortDir = "desc";

function fmtTime(sec) {
  const s = Math.max(0, Math.floor(Number(sec) || 0));
  const m = Math.floor(s / 60);
  const r = s % 60;
  return `${m}:${String(r).padStart(2, "0")}`;
}

/** Format candidate.created_at — set when scrape/scoring inserts the hit. */
function fmtScoredAt(sec) {
  const n = Number(sec);
  if (!Number.isFinite(n) || n <= 0) return "—";
  try {
    return new Date(n * 1000).toLocaleString(undefined, {
      year: "numeric",
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return "—";
  }
}

function parseReviewSort(raw) {
  const parts = String(raw || "scored:desc").split(":");
  // "discovered" kept as alias for older bookmarks / cached HTML.
  const key = parts[0] === "score" ? "score" : "scored";
  const dir = parts[1] === "asc" ? "asc" : "desc";
  return { key, dir };
}

function hitCreatedAt(h) {
  const n = Number(h?.created_at);
  return Number.isFinite(n) ? n : 0;
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

/** Turn slug ids into readable titles. */
function prettyTitle(videoId) {
  let t = String(videoId || "Untitled clip")
    .replace(/[_]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  // Soft-split common archive suffixes
  t = t.replace(/\b(20th century hall of fame)\b/i, "· $1");
  t = t.replace(/\b(archive highlights)\b/i, "· $1");
  t = t.replace(/\b(a chronicle)\b/i, "· $1");
  t = t.replace(/\s*·\s*/g, " · ");
  return t.replace(/\b\w/g, (ch) => ch.toUpperCase());
}

function youtubeId(url) {
  const u = String(url || "");
  const m =
    u.match(/[?&]v=([\w-]{6,})/) ||
    u.match(/youtu\.be\/([\w-]{6,})/) ||
    u.match(/youtube\.com\/shorts\/([\w-]{6,})/);
  return m ? m[1] : "";
}

function sourceWatchUrl(url, startSec) {
  const id = youtubeId(url);
  if (!id) return url || "";
  const t = Math.max(0, Math.floor(Number(startSec) || 0));
  return `https://www.youtube.com/watch?v=${id}&t=${t}s`;
}

/** Short plain-English label for CLIP cue text. */
function friendlyCue(cue) {
  const c = String(cue || "").trim();
  if (!c) return "Possible match";
  const low = c.toLowerCase();
  if (low.includes("shtreimel") || low.includes("streimel"))
    return "Looks like a shtreimel (fur hat)";
  if (low.includes("yarmulke") || low.includes("kippah"))
    return "Looks like a yarmulke / black hat with payot";
  if (low.includes("kapote"))
    return "Looks like Orthodox dress (coat + hat)";
  if (low.includes("rebbe"))
    return "Looks like a rebbe / elder with traditional dress";
  if (low.includes("litvish") || low.includes("yeshiva"))
    return "Looks like Litvish / yeshiva dress";
  if (low.includes("group") && (low.includes("hasidic") || low.includes("payot")))
    return "Group that looks traditionally Orthodox";
  if (low.includes("payot") || low.includes("sidelock")) {
    const covered =
      low.includes("hat") ||
      low.includes("yarmulke") ||
      low.includes("kippah") ||
      low.includes("shtreimel") ||
      low.includes("streimel");
    return covered
      ? "Looks like payot with a head covering"
      : "Sidelocks only (no clear hat / yarmulke) — usually a miss";
  }
  if (low.includes("hasidic") || low.includes("orthodox"))
    return "Looks traditionally Orthodox";
  // Fallback: trim long model prompt
  if (c.length > 72) return c.slice(0, 69) + "…";
  return c;
}

function decisionLabel(decision) {
  if (decision === "accept") return "Accepted";
  if (decision === "reject") return "Rejected";
  return "Needs review";
}

function scoreTone(score) {
  const s = Number(score) || 0;
  if (s >= 0.14) return "strong";
  if (s >= 0.11) return "solid";
  return "borderline";
}

function scoreLabel(score) {
  const tone = scoreTone(score);
  if (tone === "strong") return "Strong match";
  if (tone === "solid") return "Solid match";
  return "Borderline";
}

function groupByVideo(rows) {
  const map = new Map();
  for (const c of rows) {
    if (!map.has(c.video_id)) {
      map.set(c.video_id, {
        video_id: c.video_id,
        video_url: c.video_url,
        source_url: c.source_url,
        title: prettyTitle(c.video_id),
        hits: [],
      });
    }
    map.get(c.video_id).hits.push(c);
  }
  const list = [...map.values()];
  for (const src of list) {
    src.hits.sort((a, b) => a.start_sec - b.start_sec || a.rank - b.rank);
    src.hit_count = src.hits.length;
    src.pending = src.hits.filter((h) => !h.decision).length;
    src.accepted = src.hits.filter((h) => h.decision === "accept").length;
    src.rejected = src.hits.filter((h) => h.decision === "reject").length;
    src.best_rank_score = Math.max(...src.hits.map((h) => Number(h.rank_score) || 0));
    src.newest_created_at = Math.max(0, ...src.hits.map(hitCreatedAt));
    src.oldest_created_at = Math.min(
      ...src.hits.map((h) => hitCreatedAt(h) || Number.POSITIVE_INFINITY)
    );
    if (!Number.isFinite(src.oldest_created_at)) src.oldest_created_at = 0;
    src.best_cue = src.hits[0]?.best_cue || "";
  }
  return list;
}

function sortSources(list) {
  const dir = sortDir === "asc" ? 1 : -1;
  const ranked = [...list];
  ranked.sort((a, b) => {
    if (sortKey === "score") {
      const d = (Number(a.best_rank_score) || 0) - (Number(b.best_rank_score) || 0);
      if (d) return d * dir;
    } else {
      const av = sortDir === "asc" ? a.oldest_created_at : a.newest_created_at;
      const bv = sortDir === "asc" ? b.oldest_created_at : b.newest_created_at;
      const d = (Number(av) || 0) - (Number(bv) || 0);
      if (d) return d * dir;
    }
    return String(a.video_id || "").localeCompare(String(b.video_id || ""));
  });
  return ranked;
}

function filteredSources() {
  let list = sources;
  if (query) {
    const q = query.toLowerCase();
    list = list.filter(
      (s) =>
        s.video_id.toLowerCase().includes(q) ||
        (s.title || "").toLowerCase().includes(q) ||
        s.hits.some((h) => (h.best_cue || "").toLowerCase().includes(q))
    );
  }
  if (status === "pending") list = list.filter((s) => s.pending > 0);
  else if (status === "accept") list = list.filter((s) => s.accepted > 0);
  else if (status === "reject") list = list.filter((s) => s.rejected > 0);
  // status / openai filters are applied server-side via fetchCandidates()
  return sortSources(list);
}

function openaiDropReason(notes) {
  const line = String(notes || "")
    .split(/\r?\n/)
    .find((ln) => /^(openai|vlm):drop\b/i.test(ln.trim()));
  if (!line) return "";
  return line.replace(/^(openai|vlm):drop\s*/i, "").trim();
}

function openaiKeepReason(notes) {
  const line = String(notes || "")
    .split(/\r?\n/)
    .find((ln) => /^(openai|vlm):keep\b/i.test(ln.trim()));
  if (!line) return "";
  return line.replace(/^(openai|vlm):keep\s*/i, "").trim();
}

async function fetchCandidates() {
  let qs = "/api/candidates?limit=2000";
  if (status === "openai_drop") qs = "/api/candidates?limit=2000&openai=drop";
  else if (status === "openai_keep") qs = "/api/candidates?limit=2000&openai=keep";
  else if (status === "pending") qs = "/api/candidates?limit=2000&status=pending";
  else if (status === "accept") qs = "/api/candidates?limit=2000&status=accept";
  else if (status === "reject") qs = "/api/candidates?limit=2000&status=reject";
  const res = await fetch(qs);
  const data = await res.json();
  allCandidates = data.candidates || [];
  sources = groupByVideo(allCandidates);
  renderList();
}

function renderList() {
  const list = document.getElementById("list");
  const visible = filteredSources();
  if (!visible.length) {
    list.innerHTML = `<p class="empty">Nothing here yet — run a scrape, then come back.</p>`;
    return;
  }
  list.innerHTML = visible
    .map((s) => {
      let badge;
      if (status === "openai_drop")
        badge = `<span class="badge reject">OpenAI failed</span>`;
      else if (status === "openai_keep")
        badge = `<span class="badge accept">OpenAI pass</span>`;
      else if (s.pending === 0 && s.accepted > 0 && s.rejected === 0)
        badge = `<span class="badge accept">All kept</span>`;
      else if (s.pending === 0 && s.rejected > 0 && s.accepted === 0)
        badge = `<span class="badge reject">All passed over</span>`;
      else if (s.pending === 0) badge = `<span class="badge">Done</span>`;
      else badge = `<span class="badge">${s.pending} to check</span>`;

      const cueLine = friendlyCue(s.best_cue);
      const scoredAt =
        sortDir === "asc" ? s.oldest_created_at : s.newest_created_at;
      return `
      <button class="card ${s.video_id === activeVideoId ? "active" : ""}" data-video="${escapeAttr(s.video_id)}">
        <div class="meta">
          <span>${s.hit_count} moment${s.hit_count === 1 ? "" : "s"}</span>
          ${badge}
        </div>
        <div class="title">${escapeHtml(s.title)}</div>
        <div class="cue">${escapeHtml(cueLine)}</div>
        <div class="card-discovered">Scored ${escapeHtml(fmtScoredAt(scoredAt))}</div>
        <div class="card-foot">
          <span>${s.accepted} kept</span>
          <span>${s.rejected} passed</span>
          <span>${s.pending} left</span>
        </div>
      </button>`;
    })
    .join("");

  list.querySelectorAll(".card").forEach((el) => {
    el.addEventListener("click", () => selectSource(el.dataset.video, 0));
  });
}

function renderEmptyDetail() {
  document.getElementById("detail").innerHTML = `
    <div class="review-empty">
      <div class="review-empty-mark">◆</div>
      <h2>Pick a film on the left</h2>
      <p>You’ll see stills and can keep or pass each moment.</p>
    </div>`;
}

function currentSource() {
  return sources.find((s) => s.video_id === activeVideoId);
}

/** Move ±1 still across the filtered list (wraps between films). */
function stepImage(delta) {
  const visible = filteredSources();
  if (!visible.length) return;
  let si = visible.findIndex((s) => s.video_id === activeVideoId);
  if (si < 0) {
    selectSource(visible[0].video_id, 0);
    return;
  }
  let hi = hitIndex;
  hi += delta;
  while (true) {
    const src = visible[si];
    if (!src || !src.hits.length) return;
    if (hi >= 0 && hi < src.hits.length) {
      selectSource(src.video_id, hi);
      return;
    }
    if (delta > 0) {
      si += 1;
      if (si >= visible.length) return; // end of list
      hi = 0;
    } else {
      si -= 1;
      if (si < 0) return; // start of list
      hi = visible[si].hits.length - 1;
    }
  }
}

function selectSource(videoId, index = 0) {
  activeVideoId = videoId;
  const src = currentSource();
  if (!src || !src.hits.length) {
    renderList();
    renderEmptyDetail();
    return;
  }
  hitIndex = Math.max(0, Math.min(index, src.hits.length - 1));
  renderList();
  renderHit();
}

function renderHit() {
  const src = currentSource();
  if (!src) {
    renderEmptyDetail();
    return;
  }
  const c = src.hits[hitIndex];
  const detail = document.getElementById("detail");
  // Prefer local /media/sheet; skip expired litter.catbox hosts.
  const cloud = (c.image_url || "").includes("litter.catbox.moe") ? "" : (c.image_url || "");
  const img = c.contact_url || cloud;
  const strip = c.strip_url || "";
  const decision = c.decision || "";
  const cropStatus = c.crop_status || (c.crop_url ? "ready" : "none");
  const sheet = img
    ? `<img class="sheet" src="${escapeAttr(img)}" alt="Best stills from this moment" referrerpolicy="no-referrer" loading="eager" onerror="this.classList.add('broken');this.alt='Still could not load';" />`
    : `<div class="sheet placeholder">No still saved for this moment</div>`;
  const stripName = strip ? String(strip).split("/").pop() || "timeline_strip.jpg" : "";
  const stripBlock =
    decision === "accept" && strip
      ? `<a class="strip-download" href="${escapeAttr(strip)}" download="${escapeAttr(stripName)}">
           <span class="strip-download-icon">⬇</span>
           <span>
             <strong>Download timeline strip</strong>
             <small>±10s · every 0.5s · source + time on each frame</small>
           </span>
         </a>`
      : decision === "accept"
        ? `<p class="strip-pending">Timeline strip building… refresh soon, or run <code>scripts/build_keep_strips.py</code>.</p>`
        : "";

  let cropBlock = "";
  if (cropStatus === "ready") {
    cropBlock = `<div class="crop-action done">
      <span class="crop-action-label">
        <strong>Crop ready</strong>
        <small>±2s · every 0.5s · under 400KB</small>
      </span>
      <a class="btn ghost small" href="/crops">Open Crops</a>
    </div>`;
  } else if (cropStatus === "queued") {
    cropBlock = `<div class="crop-action busy">
      <button class="btn ghost" id="genCrop" type="button" disabled>Generating crop…</button>
      <a class="btn ghost small" href="/crops">Crops</a>
    </div>`;
  } else if (cropStatus === "error") {
    cropBlock = `<div class="crop-action">
      <button class="btn ghost" id="genCrop" type="button">Retry crop</button>
      <small class="crop-err">${escapeHtml(c.crop_error || "failed")}</small>
    </div>`;
  } else {
    cropBlock = `<div class="crop-action">
      <button class="btn ghost" id="genCrop" type="button">Generate crop</button>
      <small>±2s before/after · 0.5s steps · source + time watermark</small>
    </div>`;
  }

  const watch = sourceWatchUrl(c.source_url, c.start_sec);
  const decisionClass = decision === "accept" ? "accept" : decision === "reject" ? "reject" : "pending";
  const tone = scoreTone(c.rank_score);
  const cueFriendly = friendlyCue(c.best_cue);
  const openaiFail = openaiDropReason(c.notes);
  const openaiPass = openaiKeepReason(c.notes);
  const yt = youtubeId(c.source_url);

  const media = watch
    ? `<a class="watch-link" href="${escapeAttr(watch)}" target="_blank" rel="noopener">
         <span class="watch-icon">▶</span>
         <span>
           <strong>Watch at ${fmtTime(c.start_sec)}</strong>
           <small>${yt ? "Opens on YouTube" : "Opens source"} · film was cleared after scan</small>
         </span>
       </a>`
    : `<p class="watch-missing">No source link saved for this moment.</p>`;

  const visible = filteredSources();
  const srcIdx = visible.findIndex((s) => s.video_id === activeVideoId);
  const atFirst =
    srcIdx <= 0 && hitIndex <= 0;
  const atLast =
    srcIdx >= visible.length - 1 && hitIndex >= src.hits.length - 1;

  detail.innerHTML = `
    <div class="hit-stage">
      ${sheet}
      <div class="hit-overlay">
        <span class="badge ${decisionClass}">${escapeHtml(decisionLabel(decision))}</span>
        <span class="hit-step">Moment ${hitIndex + 1} of ${src.hits.length}</span>
      </div>
    </div>

    ${stripBlock}

    ${cropBlock}

    ${media}

    <header class="hit-head">
      <h2>${escapeHtml(src.title)}</h2>
      <p class="hit-cue">${escapeHtml(cueFriendly)}</p>
      ${
        openaiFail
          ? `<p class="hit-cue" style="opacity:.85">OpenAI failed: ${escapeHtml(openaiFail)}</p>`
          : openaiPass
            ? `<p class="hit-cue" style="opacity:.85">OpenAI pass: ${escapeHtml(openaiPass)}</p>`
            : ""
      }
    </header>

    <div class="score-row score-${tone}">
      <div class="score-main">
        <span class="score-num">${Number(c.rank_score).toFixed(3)}</span>
        <span class="score-tag">${scoreLabel(c.rank_score)}</span>
      </div>
      <div class="score-meta">
        <span>Rank #${c.rank}</span>
        <span>${fmtTime(c.start_sec)} – ${fmtTime(c.end_sec)}</span>
      </div>
    </div>

    <div class="facts">
      <div><span>Best frame</span><strong>${Number(c.peak_score).toFixed(3)}</strong></div>
      <div><span>Average</span><strong>${Number(c.mean_score).toFixed(3)}</strong></div>
      <div><span>Frames seen</span><strong>${c.hit_count}</strong></div>
      <div><span>Window</span><strong>${fmtTime(c.start_sec)}–${fmtTime(c.end_sec)}</strong></div>
      <div><span>Scored</span><strong>${escapeHtml(fmtScoredAt(c.created_at))}</strong></div>
    </div>

    <div class="actions hit-nav">
      <button class="btn ghost small" id="prevHit" ${atFirst ? "disabled" : ""}>← Previous</button>
      <button class="btn ghost small" id="nextHit" ${atLast ? "disabled" : ""}>Next →</button>
    </div>
    <div class="actions hit-decide">
      <button class="btn ok" data-act="accept">Keep</button>
      <button class="btn danger" data-act="reject">Pass</button>
      <button class="btn ghost small" data-act="clear">Undo</button>
    </div>
    <textarea class="notes" id="notes" placeholder="Optional note for yourself…">${escapeHtml(c.notes || "")}</textarea>
    <p class="hit-hint">Keep / Pass moves to the next moment. Keys: ↑ ↓ ← → or J / K</p>
  `;

  const prev = document.getElementById("prevHit");
  const next = document.getElementById("nextHit");
  if (prev)
    prev.addEventListener("click", () => {
      stepImage(-1);
    });
  if (next)
    next.addEventListener("click", () => {
      stepImage(1);
    });

  detail.querySelectorAll("[data-act]").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const decision = btn.dataset.act;
      const notes = document.getElementById("notes").value;
      const stayVideo = activeVideoId;
      const advance =
        decision === "accept" || decision === "reject"
          ? Math.min(hitIndex + 1, src.hits.length - 1)
          : hitIndex;
      await fetch("/api/review", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ key: c.key, decision, notes }),
      });
      await fetchCandidates();
      selectSource(stayVideo, advance);
    });
  });

  if (cropPollTimer) {
    clearInterval(cropPollTimer);
    cropPollTimer = null;
  }
  if (cropStatus === "queued") {
    const watchId = Number(c.id || c.key);
    const stayVideo = activeVideoId;
    const stayHit = hitIndex;
    cropPollTimer = setInterval(async () => {
      try {
        const res = await fetch(`/api/crops?id=${watchId}`);
        const data = await res.json();
        const st = (data.crop && data.crop.status) || "";
        if (st === "ready" || st === "error") {
          clearInterval(cropPollTimer);
          cropPollTimer = null;
          await fetchCandidates();
          if (activeVideoId === stayVideo && hitIndex === stayHit) {
            selectSource(stayVideo, stayHit);
          }
        }
      } catch (_) {}
    }, 2500);
  }

  const genCrop = document.getElementById("genCrop");
  if (genCrop && !genCrop.disabled) {
    genCrop.addEventListener("click", async () => {
      genCrop.disabled = true;
      genCrop.textContent = "Queuing…";
      try {
        const res = await fetch("/api/crops", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ id: Number(c.id || c.key) }),
        });
        const data = await res.json();
        if (!res.ok || !data.ok) {
          genCrop.disabled = false;
          genCrop.textContent = "Retry crop";
          return;
        }
        const stayVideo = activeVideoId;
        const stayHit = hitIndex;
        c.crop_status = (data.crop && data.crop.status) || "queued";
        c.crop_url = (data.crop && data.crop.crop_url) || c.crop_url || null;
        c.crop_error = (data.crop && data.crop.error) || null;
        await fetchCandidates();
        selectSource(stayVideo, stayHit);
      } catch (_) {
        genCrop.disabled = false;
        genCrop.textContent = "Retry crop";
      }
    });
  }
}

document.getElementById("statusChips").addEventListener("click", async (e) => {
  const btn = e.target.closest(".chip");
  if (!btn) return;
  const next = btn.dataset.status || "";
  // Always refetch — each chip uses a different API filter.
  status = next;
  document.querySelectorAll("#statusChips .chip").forEach((c) => c.classList.remove("active"));
  btn.classList.add("active");
  await fetchCandidates();
  const visible = filteredSources();
  if (!visible.find((s) => s.video_id === activeVideoId)) {
    if (visible.length) selectSource(visible[0].video_id, 0);
    else renderEmptyDetail();
  }
});

let searchTimer;
document.getElementById("search").addEventListener("input", (e) => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => {
    query = e.target.value.trim();
    renderList();
    const visible = filteredSources();
    if (!visible.find((s) => s.video_id === activeVideoId)) {
      if (visible.length) selectSource(visible[0].video_id, 0);
      else renderEmptyDetail();
    }
  }, 180);
});

document.getElementById("reviewSort")?.addEventListener("change", (e) => {
  const parsed = parseReviewSort(e.target.value);
  sortKey = parsed.key;
  sortDir = parsed.dir;
  renderList();
  const visible = filteredSources();
  if (!visible.length) {
    renderEmptyDetail();
    return;
  }
  if (!visible.find((s) => s.video_id === activeVideoId)) {
    selectSource(visible[0].video_id, 0);
  }
});

document.addEventListener("keydown", (e) => {
  if (e.target.matches("input, textarea")) return;
  const src = currentSource();
  if (!src) return;
  if (
    e.key === "ArrowDown" ||
    e.key === "ArrowRight" ||
    e.key === "j" ||
    e.key === "J"
  ) {
    e.preventDefault();
    stepImage(1);
  } else if (
    e.key === "ArrowUp" ||
    e.key === "ArrowLeft" ||
    e.key === "k" ||
    e.key === "K"
  ) {
    e.preventDefault();
    stepImage(-1);
  } else if (e.key === "a" || e.key === "A") {
    detailAct("accept");
  } else if (e.key === "r" || e.key === "R" || e.key === "x" || e.key === "X") {
    detailAct("reject");
  }
});

function detailAct(act) {
  const btn = document.querySelector(`#detail [data-act="${act}"]`);
  if (btn) btn.click();
}

fetchCandidates().then(() => {
  const visible = filteredSources();
  if (visible.length) selectSource(visible[0].video_id, 0);
  else renderEmptyDetail();
});
