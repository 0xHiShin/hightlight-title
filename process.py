import os, re, csv, json, subprocess, shlex
from pathlib import Path
from datetime import timedelta
from tqdm import tqdm
import pandas as pd
from faster_whisper import WhisperModel

# --------- Config ---------
VODS_DIR = Path("vods")
OUT_DIR = Path("out")
OUT_DIR.mkdir(parents=True, exist_ok=True)
ASR_MODEL_SIZE = os.getenv("ASR_MODEL_SIZE", "small")   # small|medium|large-v3
DEVICE = os.getenv("ASR_DEVICE", "cpu")                 # cpu|cuda
COMPUTE_TYPE = os.getenv("ASR_COMPUTE", "int8")         # int8|int8_float16|float16
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")       # ollama model tag
TITLE_LEN = int(os.getenv("TITLE_LEN", "70"))
CLIP_SEC_BEFORE = 6
CLIP_SEC_AFTER = 6

# Mots-clés multi-jeux (en/fr). Ajoute-en selon tes scènes.
KEYWORDS = [
    # génériques
    "ace","pentakill","quadra","triple","clutch","overtime","match point","map point",
    "headshot","one tap","noscope","teamfight","wipe","insane","unbelievable","what a",
    "goal","equalizer","overtime","buzzer","comeback",
    # français
    "but","égalisation","prolongation","c'est fait","incroyable","quelle action","ace !",
    "clutch !","triple élimination","quadra élimination","ace de","tête !","headshot !",
    # valorant/cs
    "defuse","retake","entry","op","operator","awp","spray","transfer",
    # moba
    "baron","nashor","dragon","ancient","roshan","highground","backdoor",
    # rocket league
    "double tap","flip reset","ceiling shot"
]
KEYWORDS = [k.lower() for k in KEYWORDS]

def ts(sec: float) -> str:
    return str(timedelta(seconds=round(sec)))

def run_cmd(cmd: str):
    return subprocess.run(shlex.split(cmd), capture_output=True, text=True)

def transcribe(video_path: Path):
    model = WhisperModel(ASR_MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    segments, info = model.transcribe(str(video_path), vad_filter=True, vad_parameters=dict(min_silence_duration_ms=300))
    segs = []
    for s in segments:
        segs.append(dict(start=float(s.start), end=float(s.end), text=s.text.strip()))
    return segs, info.language

def contains_keyword(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in KEYWORDS)

def merge_windows(cands, merge_gap=8.0):
    if not cands: return []
    cands.sort(key=lambda x: x["start"])
    merged = [cands[0].copy()]
    for c in cands[1:]:
        if c["start"] - merged[-1]["end"] <= merge_gap:
            merged[-1]["end"] = max(merged[-1]["end"], c["end"])
        else:
            merged.append(c.copy())
    return merged

def pick_context(segments, win):
    start = max(0.0, win["start"] - CLIP_SEC_BEFORE)
    end   = win["end"] + CLIP_SEC_AFTER
    ctx = " ".join(s["text"] for s in segments if s["start"] >= start and s["end"] <= end).strip()
    return start, end, ctx

def generate_title_ollama(context: str, lang_hint: str):
    sys = (
        "You generate short, punchy notification titles for live esports. "
        f"Return ONE title in {lang_hint} with <= {TITLE_LEN} characters. "
        "No hashtags. No emojis. No trailing punctuation."
    )
    prompt = f"Transcript:\n\"\"\"\n{context}\n\"\"\"\nTitle:"
    # Ollama simple prompt
    proc = subprocess.run(["ollama","run",LLM_MODEL], input=f"{sys}\n\n{prompt}", text=True, capture_output=True)
    out = proc.stdout.strip()
    # Garde la première ligne non vide
    for line in out.splitlines():
        line=line.strip().strip('"').strip()
        if line:
            return line[:TITLE_LEN]
    return out[:TITLE_LEN] if out else "Action décisive"

def cut_clip(video_path: Path, start: float, end: float, out_dir: Path):
    dur = max(1.0, end - start)
    out_file = out_dir / f"{video_path.stem}_{int(start)}_{int(end)}.mp4"
    cmd = (
        f'ffmpeg -y -ss {start:.2f} -i "{video_path}" -t {dur:.2f} '
        f'-c:v libx264 -preset veryfast -crf 23 -c:a aac -movflags +faststart "{out_file}"'
    )
    run_cmd(cmd)
    return out_file

def main():
    videos = [p for p in VODS_DIR.iterdir() if p.suffix.lower() in {".mp4",".mkv",".mov",".m4v",".webm"}]
    if not videos:
        print("Aucune VOD trouvée dans ./vods")
        return
    rows = []
    for vid in videos:
        print(f"==> Transcription: {vid.name}")
        segs, lang = transcribe(vid)
        # candidats par mots-clés
        cands = []
        for s in segs:
            if contains_keyword(s["text"]):
                cands.append({"start": s["start"], "end": s["end"]})
        # fallback si zéro candidat: prends les 5 segments les plus longs
        if not cands and segs:
            top = sorted(segs, key=lambda x: (x["end"]-x["start"]), reverse=True)[:5]
            cands = [{"start": s["start"], "end": s["end"]} for s in top]
        wins = merge_windows(cands)
        print(f"   Moments détectés: {len(wins)}")
        events_dir = OUT_DIR / "clips"
        events_dir.mkdir(exist_ok=True, parents=True)
        for i, w in enumerate(wins, 1):
            s, e, ctx = pick_context(segs, w)
            title = generate_title_ollama(ctx, "French" if lang.startswith("fr") else "English")
            clip_path = cut_clip(vid, s, e, events_dir)
            rows.append({
                "video": vid.name,
                "t_start": round(s,2),
                "t_end": round(e,2),
                "ts_window": f"{ts(s)} - {ts(e)}",
                "lang": lang,
                "title": title,
                "context": ctx,
                "clip_file": clip_path.name
            })
    # Sauvegardes
    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "events.csv"
    json_path = OUT_DIR / "events.jsonl"
    df.to_csv(csv_path, index=False)
    with open(json_path,"w",encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False)+"\n")
    print(f"\nOK -> {csv_path}\nOK -> {json_path}\nClips -> {OUT_DIR/'clips'}")

if __name__ == "__main__":
    main()
