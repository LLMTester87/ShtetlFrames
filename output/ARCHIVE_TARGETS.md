# High-volume archives for many-video hit hunting

Ranked by **expected Hasidic/Orthodox dress yield × how many videos you can actually access**. Era focus: ~1900–1950 motion picture.

## Tier A — many titles + public pull path

### 1. Steven Spielberg Jewish Film Archive (HUJI / JFC)
- **Scale:** Catalog ~20,000 titles; **~600–2,000** free online; YouTube mirrors of prewar shorts
- **Why hits:** Largest Jewish documentary vault — prewar Poland, rebbes, diaspora communities
- **Catalog:** https://jfc.org.il/en/compilation/the-steven-spielberg-jewish-film-archive-collection/
- **Virtual cinema / HUJI:** https://en.jfa.huji.ac.il/
- **Keywords:** `Poland 1939`, `Krakow`, `Hasidic`, `rebbe`, `Jerusalem`, `Agudah`, `home movie`
- **Pull:** Stream online; download only public YouTube republishes (`yt-dlp`) when allowed

### 2. USHMM Steven Spielberg Film & Video Archive
- **Scale:** **~3,500+** streamed clips; ~1,000 hours holdings
- **Why hits:** Prewar Jewish life, Munkács wedding lineage, street Orthodox dress
- **Catalog:** https://resources.ushmm.org/film
- **Keywords:** `Jewish life before`, `Munkacs`, `amateur film`, `Orthodox`, `Poland`, `wedding`
- **Pull:** Stream first; duplicate only when public-domain / ToS permits

### 3. USC MIRC — Fox Movietone News
- **Scale:** **~23,000** newsreel titles; **8,000+** digitized online (~2,000+ hours total corp)
- **Why hits:** Agudah Vienna 1923 outtakes already proven; other Orthodox Europe coverage may exist
- **Catalog:** https://digital.library.sc.edu/collections/fox-movietone-news-the-war-years/ and MIRC MVTN search
- **Keywords:** `Agudah`, `Agudas`, `Orthodox Jew`, `Vienna`, `rabbi`, `Palestine`
- **Pull:** Institutional stream; Commons/YouTube republishes for offline scan

### 4. Internet Archive (relevance-gated)
- **Scale:** Hundreds–thousands of *candidate* movies if title/description gated
- **Why hits:** Direct download of newsreels, PD shorts, user uploads of prewar Jewish film
- **Search hub:** https://archive.org/search?query=mediatype%3Amovies
- **Keywords (must appear in title/desc):** `Jewish`, `Yiddish`, `Hasid`, `Poland`, `Warsaw`, `Krakow`, `Galicia`, `Agudah`, `Munkacs`, `shtetl`, `synagogue`, `rebbe`, `Palestine`, `Jerusalem`
- **Pull:** Direct download via `discover_archives.py` → `bulk_queue.csv` (not popularity sort)

### 5. Wikimedia Commons (video files)
- **Scale:** Dozens–low hundreds of true videos (Hasidic categories are mostly stills)
- **Why hits:** Clean licenses; Agudah 1923 Fox reel already validated
- **Search:** https://commons.wikimedia.org/w/index.php?search=filetype%3Avideo+Jewish&title=Special:MediaSearch&type=video
- **Keywords:** `filetype:video` + `Jewish Poland` / `Agudah` / `Hasidic`
- **Pull:** Direct HTTP from upload.wikimedia.org

## Tier B — high yield, fewer bulk-public files

### 6. YIVO Film Archive / polishjews.yivo.org
- **Scale:** ~75 home movies + landsmanshaft films
- **Why hits:** Highest shtetl/Hasidic street density per minute
- **Links:** https://yivo.org/film · https://polishjews.yivo.org/videos
- **Access:** Mostly institutional / digitized exhibit pages — not bulk scrape

### 7. National Center for Jewish Film (NCJF)
- **Scale:** Professional corpus (Five Cities, Vishniac Carpathians, etc.)
- **Why hits:** Highest curated per-minute yield for prewar Orthodox streets
- **Link:** https://www.jewishfilm.org/
- **Access:** Public only via YouTube/IA mirrors of specific titles

### 8. NARA / Universal Newsreels (often mirrored on IA)
- **Scale:** Large federal newsreel dumps
- **Why hits:** Sparse density; mine by event keywords (Munkács, DP camps, Palestine Orthodox)
- **Link:** https://catalog.archives.gov/

### 9. Periscope Film Jewish/Israel uploads on Internet Archive
- **Scale:** Large ephemeral set
- **Why hits:** Quantity high; many post-1950 / wrong dress era — gate year ≤1950 + keywords

## Skip for auto-bulk
- Raw `Jewish OR Yiddish` sorted by downloads (pulls bananas, Chaplin, unrelated PD)
- Nazi propaganda / “Eternal Jew” / camp atrocity reels (blocklist in discovery)

## Recommended scan order (many videos, then density)
1. Seed packs (Agudah + Munkács + Five Cities mirrors) — small but guaranteed hits  
2. Relevance-gated IA bulk queue (20–40 items)  
3. Commons video hits  
4. Human browse Spielberg / USHMM / MIRC catalogs for next downloads  

## Tooling
```text
python src/discover_archives.py          # builds output/bulk_queue.csv (relevance-gated)
python src/discover_ia.py                # optional IA-only sweep (legacy; prefer discover_archives)
python src/download.py                   # pulls direct URLs from seed CSV
```
