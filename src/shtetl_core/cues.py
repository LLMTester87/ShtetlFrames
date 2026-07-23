"""Text prompts and numeric gates for Orthodox-dress zero-shot scoring."""

from __future__ import annotations

# Positives: Orthodox / Hasidic / Litvish dress cues. Prefer hat+payot+coat together.
# Bare payot alone is not enough — HEADCOVER_PROMPTS + MIN_HEADCOVER_SCORE gate that.
POSITIVE_PROMPTS = [
    "Hasidic Jewish man with beard sidelocks payot and black hat or shtreimel",
    "Hasidic Jewish man wearing a large round shtreimel fur hat and black coat",
    "Orthodox Jewish man with beard sidelocks black hat and long kapote or rekel coat",
    "Hasidic rebbe with white beard sidelocks and shtreimel fur hat",
    "group of Hasidic Jewish men in black coats streimels hats and payot",
    "Litvish yeshiva man with beard black hat and dark frock coat",
    "elderly Orthodox Jewish rabbi with long white beard black hat and dark coat",
    "Orthodox Jewish man with curled payot sidelocks and black yarmulke or fedora",
]

# Must also fire for a hit: visible Jewish/Orthodox head covering on the person.
HEADCOVER_PROMPTS = [
    "Jewish man wearing a black fedora homburg or Borsalino hat",
    "Jewish man wearing a large round shtreimel or spodik fur hat",
    "Jewish man wearing a black yarmulke kippah skullcap on his head",
    "Orthodox Jewish man with his head covered by a black hat",
    "Hasidic man in a wide brim black hat covering his head",
]

# Must look male (adult man). Compared against FEMALE_PROMPTS so women are rejected.
MALE_PROMPTS = [
    "adult man male person",
    "grown man with male face and male clothing",
    "photograph of a man not a woman",
    "male person upper body",
]

FEMALE_PROMPTS = [
    "adult woman female person",
    "grown woman with female face",
    "photograph of a woman not a man",
    "girl or young woman",
]

# Must show enough person / upper-body shape — not a tight face-only crop.
BODY_PROMPTS = [
    "upper body of a person showing shoulders and chest",
    "person from the chest up with torso and shoulders visible",
    "man wearing a coat with shoulders and upper body in frame",
    "full upper-body portrait including head shoulders and chest",
]

FACE_ONLY_PROMPTS = [
    "extreme close-up of a face only filling the frame",
    "tight headshot face crop with no shoulders visible",
    "face portrait cropped above the neck only",
    "close-up facial photo with no torso or coat visible",
]

# Hard negatives for common false positives in newsreels / Pathé / docs.
NEGATIVE_PROMPTS = [
    "modern business suit and necktie",
    "military uniform and helmet",
    "woman in modern dress",
    "adult woman or girl",
    "blurry crowd of anonymous people",
    "child only no adult man",
    "bare headed clean shaven modern man",
    "bareheaded man with curly hair or sidelocks no hat no yarmulke",
    "man with long curled hair beside ears but uncovered bare head",
    "sports jersey athletic clothing or tracksuit",
    "english gentleman in bowler hat or top hat",
    "man in fedora trilby or homburg hat no sidelocks no payot",
    "victorian or edwardian european man in dark coat and hat",
    "1950s man in overcoat and fedora without Jewish sidelocks",
    "newsreel politician or diplomat in dark overcoat",
    "astronaut space suit or NASA flight gear",
    "police officer or firefighter uniform",
    "catholic priest clerical collar",
    "christian bishop wearing a mitre or white pointed ceremonial hat",
    "eastern orthodox priest in vestments or kamilavka",
    "judge or barrister wearing a powdered wig",
    "muslim man in turban or keffiyeh",
    "sikh man wearing a turban",
    "cowboy hat western clothing",
    "bald or short hair man without beard",
    "film actor or celebrity portrait",
    "close-up face only no body or shoulders",
    "tight headshot with no torso visible",
    "secular european crowd in dark coats at a ceremony",
    # Pathé false-keep clusters (OpenAI was inventing shtreimels on these).
    "english public school boys in school uniforms and caps",
    "cricket players in white flannels and sports caps",
    "garden party society guests in hats and coats",
    "royal or aristocratic outdoor garden reception",
    "space race astronaut or rocket launch crowd",
    "british newsreel crowd of secular men in overcoats",
]

# OpenCLIP encoder (ViT-L-14 >> classic OpenAI ViT-B/32 for fine-grained dress cues).
CLIP_MODEL = "ViT-L-14"
CLIP_PRETRAINED = "laion2b_s32b_b82k"
YOLO_WEIGHTS = "yolov8s.pt"

# CLIP pre-filter before vision verify. Soft -0.28 flooded pods (~6–20 segs/video).
# 0.08 missed clear kippah Pathé (peak ~0.04); 0.04 + softer negs recovers those; OpenAI still gates.
DEFAULT_SCORE_THRESHOLD = 0.04
MIN_POS_SCORE = 0.20
MIN_HEADCOVER_SCORE = 0.16
MIN_MALE_SCORE = 0.18
MIN_BODY_SCORE = 0.16
# Slightly stricter than 0.95 — Pathé secular coats were sneaking past.
MAX_NEG_TO_POS_RATIO = 0.90
NEG_SCORE_WEIGHT = 0.90
DEFAULT_FPS = 1.5
MIN_SEGMENT_SEC = 3.0
MAX_GAP_SEC = 2.0
# Hard cap per video after CLIP grouping — stops 20× OpenAI verifies on one reel.
MAX_SEGMENTS_PER_VIDEO = 3
MIN_PERSON_AREA = 40 * 80
# Person box must be taller than wide (rejects face-square / head-only boxes).
MIN_PERSON_ASPECT = 1.15
# Absolute min bbox height in px — tiny face crops fail even if area clears.
MIN_PERSON_HEIGHT = 100
YOLO_CONF = 0.32
TOP_K_CUES = 1
TOP_K_NEGS = 3
