# Ranked public sources (high potential)

## Scan these first (known / proven yield)
1. **USC MIRC / Commons — Agudah Vienna 1923** — Fox Movietone outtakes; dense Orthodox leadership. Local: `data/videos/agudah_1923_commons.webm`
2. **NARA / public YouTube — Munkács 1933** — wedding crowds + Minchas Elazar. Local: `data/videos/munkacs_1933_yt.mp4`
3. **NCJF “Five Cities” (1938–39)** — Kraków/Lwów/Warsaw/Vilna/Białystok; chase public YouTube/IA mirrors (catalog pointers in `data/seed_videos.csv`)
4. **USHMM film catalog streams** — prewar Jewish life; stream-first, download only when public-domain permitted
5. **Spielberg Jewish Film Archive / JFC** — rebbes in Jerusalem 1930s etc. (stream catalog)

## Batch discovery pools already queried
See `output/ia_batch_discoveries.csv` (80 Internet Archive movie items, 1900–1950 keywords).  
Items flagged `blocked_from_autodownload=true` are Nazi propaganda / camp atrocity titles — **do not** auto-ingest; leave for specialist historical review only.

## Local scan completed this run
- 10 videos scanned
- 246 candidate segments → `output/candidates.jsonl` + `output/review_queue.csv`
- Contact sheets for top 40 → `output/contact_sheets/`
- Calibration threshold recommended: **0.05** (`output/calibration_report.json`)
